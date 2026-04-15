"""
SAE feature extraction via Gemma Scope 2.

For each prompt, runs a forward pass through the model and records SAE feature
activations at every layer (or a specified subset). Activations can be cached
to disk to avoid redundant forward passes across experiments.

The key token position for analysis is the final token of the prompt (the position
at which the model must produce its output), following standard interpretability
practice. For CoT analysis, we also extract activations at the end of the CoT
sequence before the final answer token.
"""

from __future__ import annotations

import gzip
import logging
import pickle
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.gemma_scope import JumpReLUSAE, load_all_layer_saes

logger = logging.getLogger(__name__)

TokenPosition = Literal["last", "answer_start", "all"]


class FeatureExtractor:
    """
    Extracts SAE feature activations for a list of prompts.

    For each prompt and each layer, computes the SAE encoding of the residual
    stream at the specified token position.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        saes: dict[int, JumpReLUSAE],
        device: str = "cuda",
        hook_point: str = "resid_post",
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.saes = saes
        self.device = device
        self.hook_point = hook_point
        self._residuals: dict[int, torch.Tensor] = {}

    def extract(
        self,
        prompts: list[str],
        token_position: TokenPosition = "last",
        batch_size: int = 8,
    ) -> dict[int, torch.Tensor]:
        """
        Extract SAE feature activations for all prompts and all loaded SAE layers.

        Returns:
            Dict mapping layer index to tensor of shape (n_prompts, n_features).
        """
        all_activations: dict[int, list[torch.Tensor]] = {layer: [] for layer in self.saes}

        for batch_start in range(0, len(prompts), batch_size):
            batch = prompts[batch_start : batch_start + batch_size]
            residuals = self._get_residuals(batch, token_position)

            for layer_idx, sae in self.saes.items():
                layer_resid = residuals[layer_idx]  # (batch, d_model)
                with torch.no_grad():
                    features, _ = sae(layer_resid.to(sae.W_enc.device))
                all_activations[layer_idx].append(features.cpu())

        return {layer: torch.cat(acts, dim=0) for layer, acts in all_activations.items()}

    def _get_residuals(
        self,
        prompts: list[str],
        token_position: TokenPosition,
    ) -> dict[int, torch.Tensor]:
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        self._residuals = {}
        hooks = self._register_hooks(inputs["input_ids"], token_position)

        with torch.no_grad():
            self.model(**inputs)

        for hook in hooks:
            hook.remove()

        return self._residuals

    def _register_hooks(
        self,
        input_ids: torch.Tensor,
        token_position: TokenPosition,
    ) -> list:
        hooks = []
        seq_len = input_ids.shape[1]

        if token_position == "last":
            positions = [-1]
        elif token_position == "all":
            positions = list(range(seq_len))
        else:
            positions = [-1]  # Default to last; answer_start requires CoT parsing

        for layer_idx in self.saes:
            layer = self.model.model.layers[layer_idx]

            def make_hook(idx: int, pos: list[int]) -> callable:
                def hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    if pos == [-1]:
                        self._residuals[idx] = hidden[:, -1, :].detach()
                    else:
                        self._residuals[idx] = hidden[:, pos, :].detach()
                return hook

            hooks.append(layer.register_forward_hook(make_hook(layer_idx, positions)))
        return hooks


def save_activations(
    activations: dict[int, torch.Tensor],
    path: str | Path,
    compress: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        with gzip.open(str(path) + ".gz", "wb") as f:
            pickle.dump(activations, f)
    else:
        with open(path, "wb") as f:
            pickle.dump(activations, f)


def load_activations(path: str | Path) -> dict[int, torch.Tensor]:
    path = Path(path)
    gz_path = Path(str(path) + ".gz")
    if gz_path.exists():
        with gzip.open(gz_path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)
