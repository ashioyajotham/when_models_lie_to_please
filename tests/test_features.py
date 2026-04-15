"""Tests for feature extraction, differential analysis, and clustering."""

import numpy as np
import pytest
import torch

from src.features.differential import (
    DifferentialFeature,
    compute_differential_activation,
    get_suppressed_features,
    layer_summary,
)
from src.features.clustering import (
    build_layer_profiles,
    cluster_features,
    _classify_layer_profile,
    LAYER_RANGES_4B,
)


class TestDifferentialActivation:
    def _make_activations(self, n_prompts: int, n_features: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        return torch.tensor(rng.random((n_prompts, n_features)), dtype=torch.float32)

    def test_detects_significant_features(self):
        # Control: feature 5 activates at ~0, treatment: feature 5 activates at ~1
        n, d = 50, 100
        control = {0: self._make_activations(n, d)}
        treatment = {0: self._make_activations(n, d)}

        # Artificially boost feature 5 in treatment
        treatment[0][:, 5] += 3.0

        result = compute_differential_activation(control, treatment, alpha=0.05)
        feature_indices = [f.feature_index for f in result[0]]
        assert 5 in feature_indices, "Feature 5 should be detected as significant."

    def test_no_false_positives_on_identical(self):
        n, d = 50, 100
        acts = {0: self._make_activations(n, d)}
        result = compute_differential_activation(acts, acts, alpha=0.05)
        # Identical distributions should yield no significant features
        assert len(result[0]) == 0, "No features should be significant when distributions are identical."

    def test_top_k_limit(self):
        n, d = 50, 200
        control = {0: self._make_activations(n, d)}
        treatment = {0: control[0] + 2.0}  # All features shifted by 2

        result = compute_differential_activation(control, treatment, alpha=0.05, top_k=10)
        assert len(result[0]) <= 10

    def test_get_suppressed_features(self):
        features = [
            DifferentialFeature(0, 0, -0.5, -5.0, 0.001, 0.001, 1.0, 0.5, -1.0),
            DifferentialFeature(1, 0, 0.5, 5.0, 0.001, 0.001, 0.5, 1.0, 1.0),
        ]
        suppressed = get_suppressed_features({0: features}, "suppressed")
        assert len(suppressed[0]) == 1
        assert suppressed[0][0].feature_index == 0

    def test_layer_summary_structure(self):
        features = [
            DifferentialFeature(0, 0, -0.5, -5.0, 0.001, 0.001, 1.0, 0.5, -1.0),
            DifferentialFeature(1, 0, 0.3, 3.0, 0.01, 0.01, 0.5, 0.8, 0.6),
        ]
        summary = layer_summary({0: features})
        assert summary[0]["n_significant"] == 2
        assert summary[0]["n_suppressed"] == 1
        assert summary[0]["n_activated"] == 1


class TestFeatureClustering:
    def _make_differential(self):
        return {
            layer: [
                DifferentialFeature(i, layer, float(i % 3 - 1), 0.0, 0.01, 0.01, 0.0, 0.0, 0.0)
                for i in range(5)
            ]
            for layer in range(26)   # 4B model
        }

    def test_build_layer_profiles_shape(self):
        differential = self._make_differential()
        matrix, feature_ids = build_layer_profiles(differential, n_layers=26)
        assert matrix.shape[1] == 26
        assert len(feature_ids) == matrix.shape[0]

    def test_classify_layer_profile(self):
        ranges = LAYER_RANGES_4B
        assert _classify_layer_profile([3, 5], ranges) == "early"
        assert _classify_layer_profile([12, 15], ranges) == "mid"
        assert _classify_layer_profile([20, 24], ranges) == "late"
        assert _classify_layer_profile([5, 15], ranges) == "multi-layer"
