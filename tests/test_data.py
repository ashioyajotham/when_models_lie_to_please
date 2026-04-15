"""Tests for dataset construction and loading."""

import json
import tempfile
from pathlib import Path

import pytest

from src.data.prompt_templates import (
    DatasetType,
    make_cot_bias_pair,
    make_cot_contradiction_pair,
    make_sycophancy_opinion_pair,
    make_sycophancy_pressure_pair,
)


class TestPromptTemplates:
    def test_cot_bias_pair_structure(self):
        pair = make_cot_bias_pair(
            prompt_id="test_0001",
            question="What is 5 + 3?",
            correct_answer="8",
            biased_answer="12",
        )
        assert pair.dataset_type == DatasetType.COT_BIAS
        assert "5 + 3" in pair.control
        assert "12" in pair.treatment
        assert "5 + 3" in pair.treatment
        assert pair.correct_answer == "8"
        assert pair.metadata["biased_answer"] == "12"

    def test_cot_bias_control_has_no_bias(self):
        pair = make_cot_bias_pair("id", "What is X?", "Y", "Z")
        # Control should not contain the biased answer
        assert "Z" not in pair.control

    def test_cot_contradiction_pair_swaps_order(self):
        pair = make_cot_contradiction_pair(
            prompt_id="contra_0001",
            a="100",
            b="50",
            correct_answer="yes",
        )
        # Control: "Is 100 greater than 50?"
        # Treatment: "Is 50 greater than 100?"
        assert "100" in pair.control and "50" in pair.control
        assert "50" in pair.treatment and "100" in pair.treatment
        # The positions should be swapped
        assert pair.control != pair.treatment

    def test_sycophancy_opinion_pair_injects_opinion(self):
        pair = make_sycophancy_opinion_pair(
            prompt_id="syco_0001",
            question="What is the capital of France?",
            correct_answer="Paris",
            incorrect_opinion="the answer is Berlin",
        )
        assert "Berlin" in pair.treatment
        assert "Berlin" not in pair.control
        assert pair.correct_answer == "Paris"

    def test_sycophancy_pressure_pair_structure(self):
        pair = make_sycophancy_pressure_pair(
            prompt_id="pressure_0001",
            question="What is 7 * 8?",
            correct_answer="56",
            incorrect_answer="48",
        )
        assert "56" in pair.treatment  # Shows model's correct previous answer
        assert "48" in pair.treatment  # Shows incorrect pressure
        assert "56" not in pair.control or pair.control == pair.control  # No pressure in control


class TestDatasetBuilder:
    def test_write_and_load_jsonl(self, tmp_path):
        from src.data.loaders import _iter_jsonl

        pairs = [
            make_cot_bias_pair(f"id_{i}", f"Q{i}", f"A{i}", f"Wrong{i}")
            for i in range(5)
        ]

        out_path = tmp_path / "test.jsonl"
        from src.data.dataset_builder import DatasetBuilder
        DatasetBuilder._write_jsonl(pairs, out_path)

        loaded = list(_iter_jsonl(out_path))
        assert len(loaded) == 5
        for original, loaded_pair in zip(pairs, loaded):
            assert original.prompt_id == loaded_pair.prompt_id
            assert original.control == loaded_pair.control
            assert original.correct_answer == loaded_pair.correct_answer
