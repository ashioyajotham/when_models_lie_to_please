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


from contextlib import contextmanager, ExitStack


@contextmanager
def apply_clamping_hook(
    model: AutoModelForCausalLM,
    layer: int,
    sae: JumpReLUSAE,
    feature_indices: list[int],
    clamp_value: float = 0.0,
):
    """
    Context manager that registers a forward hook on the model layer.
    The hook intercepts the residual stream, maps it to SAE features,
    clamps the specified features, and projects the change back.
    """
    import os
    if os.environ.get("MOCK_PIPELINE") == "true":
        yield
        return

    if hasattr(model.model, "language_model"):
        target_layer = model.model.language_model.layers[layer]
    else:
        target_layer = model.model.layers[layer]

    # Ensure SAE matches device and dtype
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    sae.to(device=device, dtype=dtype)

    def hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output

        # Preserve original shape, flatten to (batch * seq_len, d_model)
        orig_shape = hidden.shape
        hidden_flat = hidden.view(-1, orig_shape[-1]).to(device=device, dtype=dtype)

        with torch.no_grad():
            # Encode hidden states to SAE feature activations
            z = sae.encode(hidden_flat)

            # Compute difference: clamp_val - current_val
            z_to_clamp = z[:, feature_indices]
            clamp_tensor = torch.full_like(z_to_clamp, clamp_value)
            delta_z = clamp_tensor - z_to_clamp

            # Project difference back to residual stream space using decoder weights
            # W_dec shape: (n_features, d_model)
            delta_hidden = delta_z @ sae.W_dec[feature_indices]

            # Patch residual stream
            hidden_patched = hidden_flat + delta_hidden
            hidden_patched = hidden_patched.view(orig_shape)

        if is_tuple:
            return (hidden_patched,) + output[1:]
        return hidden_patched

    handle = target_layer.register_forward_hook(hook)
    try:
        yield handle
    finally:
        handle.remove()


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
    Applies all clamping configs simultaneously.
    """
    # Group features and clamp values by layer
    active_saes = {cfg.layer: saes[cfg.layer] for cfg in clamp_configs if cfg.layer in saes}
    layer_to_features: dict[int, list[int]] = {}
    layer_to_value: dict[int, float] = {}

    for cfg in clamp_configs:
        if cfg.layer not in active_saes:
            logger.warning("SAE for layer %d not loaded; skipping clamp.", cfg.layer)
            continue
        layer_to_features.setdefault(cfg.layer, []).extend(cfg.feature_indices)
        layer_to_value[cfg.layer] = cfg.clamp_value

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
                    stack.enter_context(
                        apply_clamping_hook(
                            model,
                            layer,
                            sae,
                            feature_indices,
                            layer_to_value[layer],
                        )
                    )

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
