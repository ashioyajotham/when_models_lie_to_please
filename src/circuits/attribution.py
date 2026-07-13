"""
Attribution graph construction using cross-layer transcoders.

An attribution graph is a directed acyclic graph where:
  - Nodes are SAE features at specific layers
  - Edges represent causal contribution: feature A at layer L contributes to
    feature B at layer L' (via the cross-layer transcoder from L to L')
  - Edge weight is the transcoder-predicted contribution magnitude

Construction procedure (following Anthropic Circuit Tracing, 2025):
  1. Start from the output features of interest (override features from Phase 1)
  2. For each output feature, run the cross-layer transcoder backwards to identify
     which input features at earlier layers causally contribute
  3. Recurse until reaching the input embedding layer or a contribution threshold

The result is a graph that traces the computational pathway from internal
correctness representation to override output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx
import torch

from src.utils.gemma_scope import CrossLayerTranscoder, JumpReLUSAE

logger = logging.getLogger(__name__)


@dataclass
class AttributionNode:
    layer: int
    feature_index: int
    activation: float
    label: str = ""             # Populated from feature characterization


@dataclass
class AttributionEdge:
    source: AttributionNode
    target: AttributionNode
    weight: float               # Transcoder-predicted contribution magnitude


@dataclass
class AttributionGraph:
    nodes: list[AttributionNode] = field(default_factory=list)
    edges: list[AttributionEdge] = field(default_factory=list)
    prompt_id: str = ""
    condition: str = ""         # "control", "cot_unfaithful", "sycophancy"

    def to_networkx(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(
                (node.layer, node.feature_index),
                layer=node.layer,
                feature_index=node.feature_index,
                activation=node.activation,
                label=node.label,
            )
        for edge in self.edges:
            # Ensure edge endpoint nodes exist with full attributes
            # (add_edge auto-creates bare nodes without attributes)
            src_id = (edge.source.layer, edge.source.feature_index)
            tgt_id = (edge.target.layer, edge.target.feature_index)
            if src_id not in G:
                G.add_node(
                    src_id,
                    layer=edge.source.layer,
                    feature_index=edge.source.feature_index,
                    activation=edge.source.activation,
                    label=edge.source.label,
                )
            if tgt_id not in G:
                G.add_node(
                    tgt_id,
                    layer=edge.target.layer,
                    feature_index=edge.target.feature_index,
                    activation=edge.target.activation,
                    label=edge.target.label,
                )
            G.add_edge(src_id, tgt_id, weight=edge.weight)
        return G


class AttributionGraphBuilder:
    """
    Constructs attribution graphs for specific prompts using cross-layer transcoders.

    The builder starts from a set of seed features (e.g., override features identified
    in Phase 1) and traces their causal ancestors through the model.
    """

    def __init__(
        self,
        saes: dict[int, JumpReLUSAE],
        transcoders: dict[tuple[int, int], CrossLayerTranscoder],
        edge_threshold: float = 0.05,
        max_hops: int = 10,
    ) -> None:
        self.saes = saes
        self.transcoders = transcoders
        self.edge_threshold = edge_threshold
        self.max_hops = max_hops

    def build(
        self,
        feature_activations: dict[int, torch.Tensor],
        seed_features: list[tuple[int, int]],
        prompt_id: str = "",
        condition: str = "",
    ) -> AttributionGraph:
        """
        Build an attribution graph starting from seed features.

        Args:
            feature_activations: Dict mapping layer → (n_features,) activation tensor
                                  for a single prompt (not batched).
            seed_features: List of (layer, feature_index) tuples to start from.
            prompt_id: For logging.
            condition: "control", "cot_unfaithful", or "sycophancy".

        Returns:
            AttributionGraph with nodes and edges traced up to max_hops.
        """
        graph = AttributionGraph(prompt_id=prompt_id, condition=condition)
        visited: set[tuple[int, int]] = set()
        queue = list(seed_features)

        for _ in range(self.max_hops):
            if not queue:
                break
            current_batch = queue[:]
            queue = []

            for layer, feat_idx in current_batch:
                node_id = (layer, feat_idx)
                if node_id in visited:
                    continue
                visited.add(node_id)

                activation = float(feature_activations[layer][feat_idx])
                node = AttributionNode(
                    layer=layer, feature_index=feat_idx, activation=activation
                )
                graph.nodes.append(node)

                ancestors = self._find_ancestors(
                    layer, feat_idx, feature_activations
                )
                for anc_layer, anc_feat_idx, weight in ancestors:
                    anc_activation = float(feature_activations[anc_layer][anc_feat_idx])
                    anc_node = AttributionNode(
                        layer=anc_layer, feature_index=anc_feat_idx, activation=anc_activation
                    )
                    graph.edges.append(
                        AttributionEdge(source=anc_node, target=node, weight=weight)
                    )
                    if (anc_layer, anc_feat_idx) not in visited:
                        queue.append((anc_layer, anc_feat_idx))

        return graph

    def _find_ancestors(
        self,
        target_layer: int,
        target_feat_idx: int,
        feature_activations: dict[int, torch.Tensor],
    ) -> list[tuple[int, int, float]]:
        """
        Find upstream features (layer, feat_idx, weight) that contribute to
        target_feat_idx at target_layer, using available transcoders.
        """
        ancestors = []
        for (source_layer, dest_layer), transcoder in self.transcoders.items():
            if dest_layer != target_layer:
                continue
            if source_layer not in feature_activations:
                continue

            src_activations = feature_activations[source_layer]  # (n_features,)

            if hasattr(transcoder, "W") and transcoder.W is not None:
                # Legacy / Mock transcoder
                contributions = transcoder.forward(src_activations.unsqueeze(0)).squeeze(0)
                if target_feat_idx >= contributions.shape[0]:
                    continue
                W_col = transcoder.W[:, target_feat_idx]
            else:
                # Real Gemma Scope 2 JumpReLU transcoder
                if source_layer not in self.saes or target_layer not in self.saes:
                    continue
                sae_source = self.saes[source_layer]
                sae_target = self.saes[target_layer]
                
                if target_feat_idx >= sae_target.W_enc.shape[1]:
                    continue
                
                # Dynamic right-to-left projection:
                # W_eff_col = SAE_source.W_dec @ transcoder.w_enc @ transcoder.w_dec @ SAE_target.W_enc[:, target_feat_idx]
                v0 = sae_target.W_enc[:, target_feat_idx]
                v1 = transcoder.w_dec @ v0
                v2 = transcoder.w_enc @ v1
                W_col = sae_source.W_dec @ v2

            # To attribute per-source-feature, we use the transcoder weight matrix
            W_col = W_col.to(device=src_activations.device, dtype=src_activations.dtype)
            per_source = src_activations * W_col  # (n_src_features,)

            significant = torch.where(per_source.abs() > self.edge_threshold)[0]
            for src_feat_idx in significant.tolist():
                weight = float(per_source[src_feat_idx])
                ancestors.append((source_layer, src_feat_idx, weight))

        return ancestors
