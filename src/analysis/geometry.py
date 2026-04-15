"""
Representational geometry analysis.

Computes the geometric relationship between the "unfaithfulness direction" and
the "sycophancy direction" in activation space at each layer.

The unfaithfulness direction is the mean difference vector:
    d_unfaithful = mean(activations_unfaithful) - mean(activations_faithful)

Similarly for sycophancy. If these vectors are aligned (high cosine similarity),
it suggests the same representational axis is being used for both failure modes,
providing evidence for a shared mechanism.

We also compute subspace overlap (canonical angles between subspaces spanned by
the top principal components of each condition's activation differences).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import subspace_angles

logger = logging.getLogger(__name__)


@dataclass
class LayerGeometry:
    layer: int
    cosine_similarity: float         # Between unfaithfulness and sycophancy directions
    mean_canonical_angle_deg: float  # Mean canonical angle between top-k subspaces
    min_canonical_angle_deg: float   # Smallest canonical angle (most aligned component)
    unfaithfulness_norm: float       # L2 norm of unfaithfulness direction
    sycophancy_norm: float           # L2 norm of sycophancy direction


def compute_direction(
    condition_activations: torch.Tensor,
    baseline_activations: torch.Tensor,
) -> np.ndarray:
    """
    Compute the mean difference direction between two conditions.

    Args:
        condition_activations: (n_prompts, d_features) tensor for the target condition
        baseline_activations: (n_prompts, d_features) tensor for the baseline

    Returns:
        Unit-norm direction vector of shape (d_features,)
    """
    diff = condition_activations.float().mean(0) - baseline_activations.float().mean(0)
    diff_np = diff.numpy()
    norm = np.linalg.norm(diff_np)
    if norm < 1e-10:
        return diff_np
    return diff_np / norm


def compute_layer_geometry(
    faithful_activations: dict[int, torch.Tensor],
    unfaithful_activations: dict[int, torch.Tensor],
    non_sycophantic_activations: dict[int, torch.Tensor],
    sycophantic_activations: dict[int, torch.Tensor],
    subspace_dims: int = 10,
) -> list[LayerGeometry]:
    """
    Compute geometric relationship at each layer between the two failure modes.

    Args:
        subspace_dims: Number of PCA dimensions to use for subspace angle computation.

    Returns:
        List of LayerGeometry objects, one per layer.
    """
    assert set(faithful_activations.keys()) == set(non_sycophantic_activations.keys()), \
        "Layer sets must match across all conditions."

    results = []
    for layer in sorted(faithful_activations.keys()):
        d_unfaithful = compute_direction(
            unfaithful_activations[layer], faithful_activations[layer]
        )
        d_sycophancy = compute_direction(
            sycophantic_activations[layer], non_sycophantic_activations[layer]
        )

        cosine_sim = float(
            np.dot(d_unfaithful, d_sycophancy) /
            (np.linalg.norm(d_unfaithful) * np.linalg.norm(d_sycophancy) + 1e-10)
        )

        # Subspace angles between top-k PCA components of each condition
        subspace_dim = min(subspace_dims, unfaithful_activations[layer].shape[0] - 1)
        angles_deg = _compute_subspace_angles(
            unfaithful_activations[layer].float().numpy(),
            faithful_activations[layer].float().numpy(),
            sycophantic_activations[layer].float().numpy(),
            non_sycophantic_activations[layer].float().numpy(),
            n_dims=subspace_dim,
        )

        results.append(
            LayerGeometry(
                layer=layer,
                cosine_similarity=cosine_sim,
                mean_canonical_angle_deg=float(np.mean(angles_deg)) if len(angles_deg) else 90.0,
                min_canonical_angle_deg=float(np.min(angles_deg)) if len(angles_deg) else 90.0,
                unfaithfulness_norm=float(np.linalg.norm(d_unfaithful)),
                sycophancy_norm=float(np.linalg.norm(d_sycophancy)),
            )
        )

        logger.debug(
            "Layer %d: cosine_sim=%.3f, mean_angle=%.1f°",
            layer, cosine_sim, results[-1].mean_canonical_angle_deg,
        )

    return results


def _compute_subspace_angles(
    unfaithful: np.ndarray,
    faithful: np.ndarray,
    sycophantic: np.ndarray,
    non_sycophantic: np.ndarray,
    n_dims: int,
) -> np.ndarray:
    """Compute canonical angles between the top-n_dims PCA subspaces."""
    from sklearn.decomposition import PCA

    if n_dims < 1:
        return np.array([90.0])

    try:
        pca_u = PCA(n_components=n_dims)
        pca_s = PCA(n_components=n_dims)

        U = pca_u.fit_transform(unfaithful - faithful)   # Difference matrix
        S = pca_s.fit_transform(sycophantic - non_sycophantic)

        # Canonical angles between the column spaces
        angles_rad = subspace_angles(
            pca_u.components_.T, pca_s.components_.T
        )
        return np.degrees(angles_rad)
    except Exception as exc:
        logger.warning("Subspace angle computation failed: %s", exc)
        return np.array([90.0])
