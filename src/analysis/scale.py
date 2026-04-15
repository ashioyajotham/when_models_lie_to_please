"""
Cross-scale analysis: how the override mechanism evolves from 1B to 27B parameters.

Research questions addressed here (RQ4):
  - Do the same differential features emerge at all scales?
  - Does the override circuit become more sophisticated at larger scales?
  - At what scale do sycophancy and CoT unfaithfulness first become distinguishable?

Analysis proceeds by running Phase 1 (feature discovery) on each model independently
and then comparing the resulting feature catalogs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.features.differential import DifferentialFeature

logger = logging.getLogger(__name__)

MODELS = ["gemma3_1b", "gemma3_4b", "gemma3_12b", "gemma3_27b"]
MODEL_PARAM_COUNTS = {
    "gemma3_1b": 1e9,
    "gemma3_4b": 4e9,
    "gemma3_12b": 12e9,
    "gemma3_27b": 27e9,
}
MODEL_N_LAYERS = {
    "gemma3_1b": 18,
    "gemma3_4b": 26,
    "gemma3_12b": 46,
    "gemma3_27b": 62,
}


@dataclass
class ScaleComparison:
    model_name: str
    n_significant_features: dict[int, int]   # layer → count
    peak_differential_layer: int             # Layer with most differential features
    peak_layer_fraction: float               # Peak layer / n_layers (normalized position)
    mean_effect_size: float
    n_correctness_signal_features: int       # Features labeled "correctness_signal"
    n_override_trigger_features: int         # Features labeled "override_trigger"


def normalize_layer_position(layer: int, model_name: str) -> float:
    """Convert absolute layer index to fractional position [0, 1]."""
    return layer / (MODEL_N_LAYERS[model_name] - 1)


def compare_across_scales(
    scale_results: dict[str, dict[int, list[DifferentialFeature]]],
) -> list[ScaleComparison]:
    """
    Compare Phase 1 feature discovery results across model scales.

    Args:
        scale_results: Dict mapping model_name → differential features dict
                       (as returned by compute_differential_activation)

    Returns:
        List of ScaleComparison objects, one per model, for analysis.
    """
    comparisons = []
    for model_name in MODELS:
        if model_name not in scale_results:
            logger.warning("No results for %s, skipping.", model_name)
            continue

        differential = scale_results[model_name]
        n_significant = {layer: len(features) for layer, features in differential.items()}

        if not n_significant or all(v == 0 for v in n_significant.values()):
            logger.warning("No significant features for %s.", model_name)
            continue

        peak_layer = max(n_significant, key=n_significant.get)
        all_effects = [
            f.effect_size
            for features in differential.values()
            for f in features
        ]
        comparisons.append(
            ScaleComparison(
                model_name=model_name,
                n_significant_features=n_significant,
                peak_differential_layer=peak_layer,
                peak_layer_fraction=normalize_layer_position(peak_layer, model_name),
                mean_effect_size=float(np.mean(all_effects)) if all_effects else 0.0,
                n_correctness_signal_features=0,   # Populated from characterization results
                n_override_trigger_features=0,
            )
        )

    return comparisons


def peak_layer_trend(comparisons: list[ScaleComparison]) -> dict[str, float]:
    """
    Summarize how the normalized peak differential layer position shifts with scale.

    If peak_layer_fraction is consistently ~0.6–0.7 across scales, the override
    mechanism appears at a consistent relative depth regardless of model size.
    If it shifts toward later layers at larger scales, the mechanism becomes more
    "late-binding" at scale.
    """
    return {
        c.model_name: c.peak_layer_fraction
        for c in sorted(comparisons, key=lambda c: MODEL_PARAM_COUNTS.get(c.model_name, 0))
    }
