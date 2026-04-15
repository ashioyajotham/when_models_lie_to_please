"""
Multi-layer circuit tracing via Gemma Scope 2 transcoders.

This module orchestrates the full circuit tracing pipeline for a set of prompts:
  1. Run forward pass and extract per-layer SAE feature activations
  2. Identify the layer(s) where the correctness signal diverges
     (i.e., where control and treatment activations first differ significantly)
  3. Build attribution graphs from seed override features
  4. Aggregate graphs across prompts to find consistent circuit patterns

The "divergence layer" finding is particularly important: it pinpoints where in
the forward pass the model "decides" to override its internally correct answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from src.circuits.attribution import AttributionGraph, AttributionGraphBuilder
from src.features.differential import DifferentialFeature

logger = logging.getLogger(__name__)


@dataclass
class DivergenceResult:
    prompt_id: str
    divergence_layer: int          # Layer with largest |delta| between conditions
    divergence_magnitude: float    # |f_treatment - f_control| at divergence layer
    pre_divergence_layers: list[int]   # Layers where activations are similar
    post_divergence_layers: list[int]  # Layers where activations differ


def find_divergence_layers(
    control_activations: dict[int, torch.Tensor],
    treatment_activations: dict[int, torch.Tensor],
    seed_feature_indices: list[int],
    prompt_ids: list[str] | None = None,
) -> list[DivergenceResult]:
    """
    For each prompt, identify the layer at which the seed features first diverge
    between control and treatment conditions.

    This is the candidate layer for the override mechanism.

    Args:
        control_activations: Dict mapping layer → (n_prompts, n_features) tensor
        treatment_activations: Dict mapping layer → (n_prompts, n_features) tensor
        seed_feature_indices: Feature indices identified in Phase 1 as differential
        prompt_ids: Optional list of prompt identifiers for logging
    """
    n_prompts = next(iter(control_activations.values())).shape[0]
    if prompt_ids is None:
        prompt_ids = [f"prompt_{i}" for i in range(n_prompts)]

    layers = sorted(control_activations.keys())
    results = []

    for i in range(n_prompts):
        layer_deltas = {}
        for layer in layers:
            ctrl = control_activations[layer][i, seed_feature_indices]
            trt = treatment_activations[layer][i, seed_feature_indices]
            layer_deltas[layer] = float((trt - ctrl).abs().mean())

        divergence_layer = max(layer_deltas, key=layer_deltas.get)
        divergence_magnitude = layer_deltas[divergence_layer]

        # Threshold: layers with delta < 10% of divergence magnitude are "pre-divergence"
        threshold = 0.1 * divergence_magnitude
        pre_divergence = [l for l in layers if l < divergence_layer and layer_deltas[l] < threshold]
        post_divergence = [l for l in layers if l > divergence_layer]

        results.append(
            DivergenceResult(
                prompt_id=prompt_ids[i],
                divergence_layer=divergence_layer,
                divergence_magnitude=divergence_magnitude,
                pre_divergence_layers=pre_divergence,
                post_divergence_layers=post_divergence,
            )
        )

    return results


def aggregate_divergence_layers(
    divergence_results: list[DivergenceResult],
) -> dict[int, float]:
    """
    Count how often each layer is the divergence layer across prompts.
    Returns a dict mapping layer → frequency (fraction of prompts).
    """
    total = len(divergence_results)
    counts: dict[int, int] = {}
    for r in divergence_results:
        counts[r.divergence_layer] = counts.get(r.divergence_layer, 0) + 1
    return {layer: count / total for layer, count in counts.items()}


def trace_circuits_for_condition(
    condition_activations: dict[int, torch.Tensor],
    seed_features: list[tuple[int, int]],
    graph_builder: AttributionGraphBuilder,
    prompt_ids: list[str] | None = None,
    condition: str = "",
    n_prompts_to_trace: int = 50,
) -> list[AttributionGraph]:
    """
    Build attribution graphs for a sample of prompts under a given condition.

    Args:
        condition_activations: Dict mapping layer → (n_prompts, n_features) tensor
        seed_features: List of (layer, feature_index) to use as graph roots
        graph_builder: AttributionGraphBuilder instance with loaded transcoders
        n_prompts_to_trace: Limit tracing to this many prompts (expensive operation)
    """
    n_prompts = next(iter(condition_activations.values())).shape[0]
    n_to_trace = min(n_prompts, n_prompts_to_trace)
    if prompt_ids is None:
        prompt_ids = [f"prompt_{i}" for i in range(n_prompts)]

    graphs = []
    for i in range(n_to_trace):
        single_prompt_activations = {
            layer: acts[i] for layer, acts in condition_activations.items()
        }
        graph = graph_builder.build(
            feature_activations=single_prompt_activations,
            seed_features=seed_features,
            prompt_id=prompt_ids[i],
            condition=condition,
        )
        graphs.append(graph)
        if (i + 1) % 10 == 0:
            logger.info("Traced %d/%d prompts for condition '%s'", i + 1, n_to_trace, condition)

    return graphs


def find_consistent_edges(
    graphs: list[AttributionGraph],
    min_frequency: float = 0.3,
) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    """
    Find edges that appear in at least min_frequency of attribution graphs.

    These are the consistent circuit components — the parts of the override pathway
    that are not prompt-specific.

    Returns:
        List of (source_node_id, target_node_id, frequency) tuples.
    """
    edge_counts: dict[tuple, int] = {}
    total = len(graphs)

    for graph in graphs:
        seen_in_graph = set()
        for edge in graph.edges:
            key = (
                (edge.source.layer, edge.source.feature_index),
                (edge.target.layer, edge.target.feature_index),
            )
            if key not in seen_in_graph:
                edge_counts[key] = edge_counts.get(key, 0) + 1
                seen_in_graph.add(key)

    result = [
        (src, tgt, count / total)
        for (src, tgt), count in edge_counts.items()
        if count / total >= min_frequency
    ]
    result.sort(key=lambda x: x[2], reverse=True)
    return result
