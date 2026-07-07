"""
Feature clamping interventions.

Sets identified override features to a fixed value (typically 0.0) during
generation, preventing the override circuit from activating.

This is the simplest intervention: surgical removal of specific computational
components. If it successfully restores faithful/non-sycophantic behavior,
those features are causally necessary for the override.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.gemma_scope import JumpReLUSAE
from src.utils.mishax_wrapper import clamp_features

logger = logging.getLogger(__name__)


@dataclass
class ClampingConfig:
    layer: int
    feature_indices: list[int]
    clamp_value: float = 0.0
    label: str = ""   # Human-readable label for logging


def generate_with_clamping(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    saes: dict[int, JumpReLUSAE],
    prompts: list[str],
    clamp_configs: list[ClampingConfig],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    batch_size: int = 16,
) -> list[str]:
    """
    Generate model outputs with specified features clamped.

    Applies all clamping configs simultaneously. Multiple configs targeting
    the same layer will be composed: the last config wins for overlapping features.
    """
    # Apply all clamps via context managers
    active_saes = {cfg.layer: saes[cfg.layer] for cfg in clamp_configs if cfg.layer in saes}
    # Group by layer
    layer_to_features: dict[int, list[int]] = {}
    layer_to_value: dict[int, float] = {}
    for cfg in clamp_configs:
        if cfg.layer not in active_saes:
            logger.warning("SAE for layer %d not loaded; skipping clamp.", cfg.layer)
            continue
        layer_to_features.setdefault(cfg.layer, []).extend(cfg.feature_indices)
        layer_to_value[cfg.layer] = cfg.clamp_value

    from contextlib import ExitStack

    outputs_list = []

    # Run generation in batches to prevent OutOfMemoryError
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(
            next(model.parameters()).device
        )

        with torch.no_grad():
            with ExitStack() as stack:
                for layer, feature_indices in layer_to_features.items():
                    sae = active_saes[layer]
                    stack.enter_context(clamp_features(sae, feature_indices, layer_to_value[layer]))

                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0),
                    temperature=temperature if temperature > 0 else None,
                )

        for j, ids in enumerate(output_ids):
            generated = tokenizer.decode(ids[inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            outputs_list.append(generated)

    return outputs_list
