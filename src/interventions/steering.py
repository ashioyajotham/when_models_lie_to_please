"""
Directional activation steering.

Adds a steering vector to the residual stream during generation to shift the model
toward or away from a target direction. Two steering methods are implemented:

  1. DiffMean (Rimsky et al., 2024 baseline):
     steering_vector = mean(activations_non_sycophantic) - mean(activations_sycophantic)
     Applied as: residual += multiplier * steering_vector

  2. SAE Feature Direction (our method):
     steering_vector = weighted sum of SAE feature decoder directions
     Weights are the differential activation magnitudes from Phase 1.
     This is more interpretable than DiffMean as it operates in SAE feature space.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.features.differential import DifferentialFeature
from src.utils.gemma_scope import JumpReLUSAE

logger = logging.getLogger(__name__)


@dataclass
class SteeringVector:
    layer: int
    direction: torch.Tensor    # Shape: (d_model,), unit norm
    multiplier: float
    method: str                # "diffmean" or "sae_feature_direction"


def compute_diffmean_vector(
    baseline_activations: torch.Tensor,
    target_activations: torch.Tensor,
) -> torch.Tensor:
    """
    Compute a DiffMean steering vector (Rimsky et al. 2024).

    The vector points from the target condition toward the baseline (faithful/non-sycophantic).
    Applying it with multiplier > 0 steers away from the target condition.

    Args:
        baseline_activations: (n, d_model) activations for baseline condition
        target_activations: (n, d_model) activations for target (unfaithful/sycophantic) condition

    Returns:
        Unit-norm direction vector, shape (d_model,)
    """
    diff = baseline_activations.float().mean(0) - target_activations.float().mean(0)
    norm = diff.norm()
    return diff / (norm + 1e-10)


def compute_sae_feature_steering_vector(
    differential_features: list[DifferentialFeature],
    sae: JumpReLUSAE,
    top_k: int = 20,
) -> torch.Tensor:
    """
    Compute a steering vector from the top-k SAE feature directions.

    The vector is the weighted sum of feature decoder directions, where weights
    are the differential activation values (negative for suppressed features,
    meaning we steer to restore their activation).

    Args:
        differential_features: From Phase 1 differential analysis, already filtered
                                for the relevant condition and layer.
        top_k: Number of top features to include.

    Returns:
        Unnormalized direction vector in d_model space, shape (d_model,)
    """
    sorted_features = sorted(differential_features, key=lambda f: abs(f.delta_activation), reverse=True)
    top_features = sorted_features[:top_k]

    d_model = sae.W_dec.shape[1]
    vector = torch.zeros(d_model)

    for feat in top_features:
        # Steer toward the faithful direction: negate the differential activation
        # (if feature is suppressed in treatment, we want to restore it)
        weight = -feat.delta_activation
        vector += weight * sae.W_dec[feat.feature_index]

    return vector


def apply_steering_hook(
    model: AutoModelForCausalLM,
    layer: int,
    steering_vector: torch.Tensor,
    multiplier: float,
) -> Callable:
    """
    Register a forward hook that adds multiplier * steering_vector to the
    residual stream at the specified layer.

    Returns the hook handle for removal.
    """
    if os.environ.get("MOCK_PIPELINE") == "true":
        from src.utils.mock_utils import set_mock_intervention_active
        set_mock_intervention_active(True)
        class DummyHandle:
            def remove(self):
                from src.utils.mock_utils import set_mock_intervention_active
                set_mock_intervention_active(False)
        return DummyHandle()

    target_layer = model.model.layers[layer]
    v = steering_vector.to(next(model.parameters()).device)

    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden + multiplier * v.unsqueeze(0).unsqueeze(0)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    return target_layer.register_forward_hook(hook)


def generate_with_steering(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    steering_vectors: list[SteeringVector],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> list[str]:
    """Generate with multiple steering vectors applied simultaneously."""
    handles = []
    for sv in steering_vectors:
        handle = apply_steering_hook(model, sv.layer, sv.direction, sv.multiplier)
        handles.append(handle)

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
        next(model.parameters()).device
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
        )

    for handle in handles:
        handle.remove()

    return [
        tokenizer.decode(ids[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for ids in output_ids
    ]
