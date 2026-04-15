"""
Wrapper around Google DeepMind's Mishax library for activation patching.

Mishax provides low-level hooks for intercepting and modifying activations during
the forward pass of transformer models. It was open-sourced alongside the Gemma
Scope 2 release and is the activation patching tool used in GDM's interpretability work.

Install: pip install mishax
Documentation: https://github.com/google-deepmind/mishax
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Callable, Generator

import torch

logger = logging.getLogger(__name__)

try:
    import mishax  # type: ignore[import]
    MISHAX_AVAILABLE = True
except ImportError:
    MISHAX_AVAILABLE = False
    logger.warning(
        "Mishax not installed. Activation patching will use manual hooks. "
        "Install with: pip install mishax"
    )


@contextmanager
def patch_activations(
    model: torch.nn.Module,
    layer_patches: dict[int, dict[str, torch.Tensor]],
) -> Generator[None, None, None]:
    """
    Context manager that patches residual stream activations at specified layers.

    Args:
        model: The transformer model (Gemma 3 via HuggingFace transformers)
        layer_patches: Dict mapping layer index to a dict of hook_point → patch tensor.
                       Patch tensors replace the activation at that hook point.

    Example:
        with patch_activations(model, {10: {"resid_post": faithful_act[10]}}):
            output = model(**inputs)
    """
    if MISHAX_AVAILABLE:
        yield from _patch_via_mishax(model, layer_patches)
    else:
        yield from _patch_via_manual_hooks(model, layer_patches)


@contextmanager
def _patch_via_mishax(
    model: torch.nn.Module,
    layer_patches: dict[int, dict[str, torch.Tensor]],
) -> Generator[None, None, None]:
    # Mishax provides a cleaner API; delegate when available.
    # The specific API calls depend on the mishax version; this is a sketch.
    with mishax.patch_model(model, layer_patches):
        yield


@contextmanager
def _patch_via_manual_hooks(
    model: torch.nn.Module,
    layer_patches: dict[int, dict[str, torch.Tensor]],
) -> Generator[None, None, None]:
    handles = []
    for layer_idx, patches in layer_patches.items():
        for hook_name, patch_tensor in patches.items():
            layer = _get_layer(model, layer_idx)
            submodule = _get_submodule(layer, hook_name)

            def make_hook(patch: torch.Tensor) -> Callable:
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        return (patch,) + output[1:]
                    return patch
                return hook

            handle = submodule.register_forward_hook(make_hook(patch_tensor))
            handles.append(handle)

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _get_layer(model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    # Gemma 3 HuggingFace naming: model.model.layers[i]
    return model.model.layers[layer_idx]


def _get_submodule(layer: torch.nn.Module, hook_name: str) -> torch.nn.Module:
    mapping = {
        "resid_post": layer,
        "mlp_out": layer.mlp,
        "attn_out": layer.self_attn,
    }
    if hook_name not in mapping:
        raise ValueError(f"Unknown hook point: {hook_name}. Expected one of {list(mapping)}")
    return mapping[hook_name]


@contextmanager
def clamp_features(
    sae: "JumpReLUSAE",  # noqa: F821
    feature_indices: list[int],
    clamp_value: float = 0.0,
) -> Generator[None, None, None]:
    """
    Context manager that clamps specified SAE features to a fixed value during encoding.

    Used for ablation studies: set override features to zero and measure
    whether faithful/non-sycophantic behavior is restored.
    """
    original_encode = sae.encode

    def clamped_encode(x: torch.Tensor) -> torch.Tensor:
        z = original_encode(x)
        z[:, feature_indices] = clamp_value
        return z

    sae.encode = clamped_encode
    try:
        yield
    finally:
        sae.encode = original_encode
