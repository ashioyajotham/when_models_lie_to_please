"""Tests for circuit tracing, attribution graphs, and validation utilities."""

import torch
import pytest

from src.circuits.attribution import AttributionGraph, AttributionNode, AttributionEdge
from src.circuits.transcoder_tracing import (
    find_divergence_layers,
    aggregate_divergence_layers,
    find_consistent_edges,
)
from src.interventions.evaluation import (
    score_faithfulness,
    score_sycophancy_rate,
    score_over_correction,
    _extract_final_answer,
    _answers_match,
)


class TestAttributionGraph:
    def test_to_networkx_empty(self):
        graph = AttributionGraph()
        G = graph.to_networkx()
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0

    def test_to_networkx_with_nodes_and_edges(self):
        graph = AttributionGraph(prompt_id="test", condition="control")
        n1 = AttributionNode(layer=5, feature_index=10, activation=1.5, label="correctness_signal")
        n2 = AttributionNode(layer=8, feature_index=20, activation=0.8, label="override_trigger")
        graph.nodes = [n1, n2]
        graph.edges = [AttributionEdge(source=n1, target=n2, weight=0.3)]

        G = graph.to_networkx()
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 1
        assert G.nodes[(5, 10)]["label"] == "correctness_signal"


class TestDivergenceAnalysis:
    def _make_activations(self, n_prompts, n_features, n_layers):
        return {
            layer: torch.randn(n_prompts, n_features)
            for layer in range(n_layers)
        }

    def test_find_divergence_layers_returns_one_per_prompt(self):
        n, d, L = 10, 50, 26
        ctrl = self._make_activations(n, d, L)
        trt = self._make_activations(n, d, L)
        # Spike treatment at layer 15 for seed feature 0
        trt[15][:, 0] += 10.0

        results = find_divergence_layers(ctrl, trt, seed_feature_indices=[0])
        assert len(results) == n
        # Most prompts should identify layer 15 as divergence
        divergence_layers = [r.divergence_layer for r in results]
        assert divergence_layers.count(15) > n // 2

    def test_aggregate_divergence_layers_sums_to_one(self):
        from src.circuits.transcoder_tracing import DivergenceResult
        results = [
            DivergenceResult("p0", 10, 1.0, [], []),
            DivergenceResult("p1", 10, 0.8, [], []),
            DivergenceResult("p2", 15, 1.2, [], []),
        ]
        agg = aggregate_divergence_layers(results)
        assert abs(sum(agg.values()) - 1.0) < 1e-6

    def test_find_consistent_edges(self):
        # Create 10 graphs, each with the same edge (5,10) → (8,20) and a unique edge
        graphs = []
        for i in range(10):
            graph = AttributionGraph()
            n1 = AttributionNode(5, 10, 1.0)
            n2 = AttributionNode(8, 20, 0.8)
            n3 = AttributionNode(3, i, 0.5)  # Unique edge per graph
            graph.nodes = [n1, n2, n3]
            graph.edges = [
                AttributionEdge(n1, n2, 0.5),
                AttributionEdge(n3, n2, 0.3),
            ]
            graphs.append(graph)

        consistent = find_consistent_edges(graphs, min_frequency=0.8)
        edge_ids = [(s, t) for s, t, _ in consistent]
        assert ((5, 10), (8, 20)) in edge_ids
        # None of the unique edges should be in the consistent set
        for i in range(10):
            assert ((3, i), (8, 20)) not in edge_ids


class TestEvaluationMetrics:
    def test_extract_final_answer_from_the_answer_is(self):
        output = "Let me think... The answer is 42."
        assert "42" in _extract_final_answer(output)

    def test_extract_final_answer_from_equals(self):
        output = "Computing: 5 + 3 = 8"
        assert "8" in _extract_final_answer(output)

    def test_score_faithfulness_perfect(self):
        outputs = ["The answer is Paris.", "The answer is 8."]
        controls = ["Paris is the capital.", "8 is the result."]
        score = score_faithfulness(outputs, controls)
        assert 0.0 <= score <= 1.0

    def test_score_sycophancy_rate_capitulation(self):
        initial = ["The answer is Paris."]
        post_pressure = ["You're right, it's London."]
        correct = ["Paris"]
        rate = score_sycophancy_rate(initial, post_pressure, correct)
        assert rate == 1.0  # Fully capitulated

    def test_score_sycophancy_rate_no_capitulation(self):
        initial = ["The answer is Paris."]
        post_pressure = ["I maintain my answer: Paris."]
        correct = ["Paris"]
        rate = score_sycophancy_rate(initial, post_pressure, correct)
        assert rate == 0.0  # Did not capitulate
