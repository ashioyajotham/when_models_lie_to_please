"""
Feature clustering across layers.

Groups differential features by their activation pattern across layers to identify:
  - Early-emerging features (layers 0–8 for 4B)
  - Mid-layer features (layers 9–17) — most likely locus of override mechanism
  - Late-layer features (layers 18–25) — output formatting and social calibration

Clustering is done hierarchically (Ward linkage) on a feature's layer activation
profile: a vector of delta_activation values across layers (0 if the feature is
not significant at a given layer).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from src.features.differential import DifferentialFeature

logger = logging.getLogger(__name__)

LAYER_RANGES_4B = {"early": (0, 8), "mid": (9, 17), "late": (18, 25)}
LAYER_RANGES_12B = {"early": (0, 15), "mid": (16, 30), "late": (31, 45)}
LAYER_RANGES_27B = {"early": (0, 20), "mid": (21, 40), "late": (41, 61)}

LAYER_RANGES = {
    "gemma3_4b": LAYER_RANGES_4B,
    "gemma3_12b": LAYER_RANGES_12B,
    "gemma3_27b": LAYER_RANGES_27B,
}


@dataclass
class FeatureCluster:
    cluster_id: int
    layer_profile: str        # "early", "mid", "late", or "multi-layer"
    feature_indices: list[tuple[int, int]]  # (layer, feature_index) pairs
    mean_delta: float
    peak_layer: int           # Layer with maximum absolute delta


def build_layer_profiles(
    differential: dict[int, list[DifferentialFeature]],
    n_layers: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Construct a matrix of shape (n_unique_features, n_layers) where each row
    represents a unique (feature_index) and each column is the delta_activation
    at that layer (0 if not significant).

    Features are identified by their index within the SAE; the same index at
    different layers may refer to different features. We treat (layer, index) pairs
    as the unit of analysis for clustering.

    Returns:
        profile_matrix: (n_features, n_layers) numpy array
        feature_ids: list of (layer, feature_index) pairs corresponding to rows
    """
    # Build a dict: feature_index → {layer: delta}
    feature_layer_map: dict[int, dict[int, float]] = {}
    for layer, features in differential.items():
        for f in features:
            if f.feature_index not in feature_layer_map:
                feature_layer_map[f.feature_index] = {}
            feature_layer_map[f.feature_index][layer] = f.delta_activation

    feature_ids = []
    rows = []
    for feature_index, layer_deltas in feature_layer_map.items():
        row = np.zeros(n_layers)
        peak_layer = -1
        for layer, delta in layer_deltas.items():
            row[layer] = delta
            if peak_layer == -1 or abs(delta) > abs(row[peak_layer]):
                peak_layer = layer
        feature_ids.append((peak_layer, feature_index))
        rows.append(row)

    if not rows:
        return np.empty((0, n_layers)), []

    return np.stack(rows), feature_ids


def cluster_features(
    differential: dict[int, list[DifferentialFeature]],
    n_layers: int,
    model_name: str = "gemma3_4b",
    linkage_method: str = "ward",
    n_clusters: int | None = None,
    distance_threshold: float = 0.5,
) -> list[FeatureCluster]:
    """
    Hierarchically cluster features by their layer activation profile.

    Args:
        n_clusters: If None, use distance_threshold to determine clusters.
        distance_threshold: Used when n_clusters is None.

    Returns:
        List of FeatureCluster objects.
    """
    profile_matrix, feature_ids = build_layer_profiles(differential, n_layers)

    if profile_matrix.shape[0] < 2:
        logger.warning("Too few features to cluster (%d).", profile_matrix.shape[0])
        return []

    # Normalize rows for cosine distance
    norms = np.linalg.norm(profile_matrix, axis=1, keepdims=True) + 1e-10
    normalized = profile_matrix / norms

    Z = linkage(normalized, method=linkage_method, metric="euclidean")

    if n_clusters is not None:
        labels = fcluster(Z, n_clusters, criterion="maxclust")
    else:
        labels = fcluster(Z, distance_threshold, criterion="distance")

    layer_ranges = LAYER_RANGES.get(model_name, LAYER_RANGES_4B)

    clusters: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(i)

    result = []
    for cluster_id, member_indices in clusters.items():
        members = [feature_ids[i] for i in member_indices]
        deltas = [
            differential[layer][
                next(j for j, f in enumerate(differential[layer]) if f.feature_index == fi)
            ].delta_activation
            for layer, fi in members
            if layer in differential
            and any(f.feature_index == fi for f in differential.get(layer, []))
        ]
        peak_layer = max(
            (layer for layer, _ in members),
            key=lambda l: abs(profile_matrix[feature_ids.index(next(m for m in members if m[0] == l)), l])
            if any(m[0] == l for m in members) else 0,
            default=0,
        )
        layer_profile = _classify_layer_profile(
            [layer for layer, _ in members], layer_ranges
        )
        result.append(
            FeatureCluster(
                cluster_id=cluster_id,
                layer_profile=layer_profile,
                feature_indices=members,
                mean_delta=float(np.mean(deltas)) if deltas else 0.0,
                peak_layer=peak_layer,
            )
        )

    result.sort(key=lambda c: abs(c.mean_delta), reverse=True)
    return result


def _classify_layer_profile(
    layers: list[int],
    layer_ranges: dict[str, tuple[int, int]],
) -> str:
    if not layers:
        return "unknown"
    in_early = any(layer_ranges["early"][0] <= l <= layer_ranges["early"][1] for l in layers)
    in_mid = any(layer_ranges["mid"][0] <= l <= layer_ranges["mid"][1] for l in layers)
    in_late = any(layer_ranges["late"][0] <= l <= layer_ranges["late"][1] for l in layers)
    zones = sum([in_early, in_mid, in_late])
    if zones > 1:
        return "multi-layer"
    if in_early:
        return "early"
    if in_mid:
        return "mid"
    return "late"
