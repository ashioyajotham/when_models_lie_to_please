"""
Differential activation analysis.

Given SAE feature activations for paired (control, treatment) prompts,
computes per-feature, per-layer differential activation and identifies
features with statistically significant differences.

The primary statistic is mean differential activation:
    delta_f = mean(f_treatment) - mean(f_control)

Significance is assessed via a two-sided Welch t-test with Bonferroni correction
over the number of features tested (typically 20k–40k per layer).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DifferentialFeature:
    feature_index: int
    layer: int
    delta_activation: float          # mean(treatment) - mean(control)
    t_statistic: float
    p_value_raw: float
    p_value_corrected: float         # Bonferroni-corrected
    mean_control: float
    mean_treatment: float
    effect_size: float               # Cohen's d


def compute_differential_activation(
    control_activations: dict[int, torch.Tensor],
    treatment_activations: dict[int, torch.Tensor],
    alpha: float = 0.05,
    top_k: int = 200,
) -> dict[int, list[DifferentialFeature]]:
    """
    Identify differentially active features per layer.

    Args:
        control_activations: Dict mapping layer → (n_prompts, n_features) tensor
        treatment_activations: Dict mapping layer → (n_prompts, n_features) tensor
        alpha: Significance threshold (after Bonferroni correction)
        top_k: Maximum number of features to return per layer

    Returns:
        Dict mapping layer index to list of DifferentialFeature objects,
        sorted by |delta_activation| descending.
    """
    assert set(control_activations.keys()) == set(treatment_activations.keys()), \
        "Layer sets must match between control and treatment activations."

    results: dict[int, list[DifferentialFeature]] = {}

    for layer in sorted(control_activations.keys()):
        ctrl = control_activations[layer].detach().cpu().float().numpy()   # (n, d)
        trt = treatment_activations[layer].detach().cpu().float().numpy()  # (n, d)

        n_features = ctrl.shape[1]
        t_stats, p_values = stats.ttest_ind(trt, ctrl, axis=0, equal_var=False)
        delta = trt.mean(axis=0) - ctrl.mean(axis=0)

        # Bonferroni correction
        p_corrected = np.minimum(p_values * n_features, 1.0)

        # Cohen's d effect size
        pooled_std = np.sqrt(
            (ctrl.std(axis=0) ** 2 + trt.std(axis=0) ** 2) / 2 + 1e-10
        )
        cohens_d = delta / pooled_std

        significant_mask = p_corrected < alpha
        significant_indices = np.where(significant_mask)[0]

        if len(significant_indices) == 0:
            logger.info("Layer %d: no significant features at alpha=%.3f", layer, alpha)
            results[layer] = []
            continue

        features = [
            DifferentialFeature(
                feature_index=int(idx),
                layer=layer,
                delta_activation=float(delta[idx]),
                t_statistic=float(t_stats[idx]),
                p_value_raw=float(p_values[idx]),
                p_value_corrected=float(p_corrected[idx]),
                mean_control=float(ctrl[:, idx].mean()),
                mean_treatment=float(trt[:, idx].mean()),
                effect_size=float(cohens_d[idx]),
            )
            for idx in significant_indices
        ]

        # Sort by absolute delta, take top_k
        features.sort(key=lambda f: abs(f.delta_activation), reverse=True)
        results[layer] = features[:top_k]

        logger.info(
            "Layer %d: %d significant features (Bonferroni α=%.3f), returning top %d",
            layer, len(features), alpha, min(len(features), top_k),
        )

    return results


def get_suppressed_features(
    differential: dict[int, list[DifferentialFeature]],
    direction: str = "suppressed",
) -> dict[int, list[DifferentialFeature]]:
    """
    Filter features by direction of differential activation.

    Args:
        direction: "suppressed" (treatment < control, i.e., delta < 0)
                   "activated"  (treatment > control, i.e., delta > 0)
                   "both"

    Returns:
        Filtered dict with same structure as input.
    """
    filtered: dict[int, list[DifferentialFeature]] = {}
    for layer, features in differential.items():
        if direction == "suppressed":
            filtered[layer] = [f for f in features if f.delta_activation < 0]
        elif direction == "activated":
            filtered[layer] = [f for f in features if f.delta_activation > 0]
        else:
            filtered[layer] = features
    return filtered


def layer_summary(differential: dict[int, list[DifferentialFeature]]) -> dict[int, dict]:
    """Compute summary statistics per layer for logging and plotting."""
    summary = {}
    for layer, features in differential.items():
        if not features:
            summary[layer] = {"n_significant": 0}
            continue
        deltas = [f.delta_activation for f in features]
        summary[layer] = {
            "n_significant": len(features),
            "max_delta": max(deltas, key=abs),
            "mean_delta": np.mean(deltas),
            "n_suppressed": sum(1 for f in features if f.delta_activation < 0),
            "n_activated": sum(1 for f in features if f.delta_activation > 0),
            "mean_effect_size": np.mean([f.effect_size for f in features]),
        }
    return summary
