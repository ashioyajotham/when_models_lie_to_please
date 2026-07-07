"""Tests for the mock classes and monkeypatching mechanisms."""

import os
import torch
import pytest

from src.utils.mock_utils import (
    MockModel,
    MockTokenizer,
    MockSAE,
    MockTranscoder,
    is_mock_intervention_active,
    set_mock_intervention_active,
)
from src.interventions.clamping import clamp_features
from src.interventions.steering import apply_steering_hook
from src.utils.mishax_wrapper import patch_activations


class TestMockComponents:
    def test_mock_tokenizer(self):
        tokenizer = MockTokenizer()
        # Mock load datasets might not load anything if pairs are empty or not generated yet,
        # but let's test encoding
        inputs = tokenizer(["Hello world", "Test sentence"])
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert inputs["input_ids"].shape == (2, 4)
        assert inputs["input_ids"][0, 0].item() == 1000
        assert inputs["input_ids"][1, 0].item() == 1001

        # Test decode
        assert tokenizer.decode([1000]) == "The answer is 42."

    def test_mock_model(self):
        model = MockModel(n_layers=5, d_model=32)
        assert len(model.model.layers) == 5
        assert next(model.parameters()).device == torch.device("cpu")

        # Test forward pass triggers hooks
        input_ids = torch.tensor([[1000, 1, 1, 1]], dtype=torch.long)
        hook_called = False

        def dummy_hook(module, inputs, output):
            nonlocal hook_called
            hook_called = True

        model.model.layers[2].register_forward_hook(dummy_hook)
        out = model(input_ids)
        assert hook_called
        assert out.logits.shape[0] == 1

        # Test generate
        gen_ids = model.generate(input_ids)
        assert gen_ids.shape == (1, 5)
        assert gen_ids[0, -1].item() == 2000

    def test_mock_sae(self):
        sae = MockSAE(d_model=32, n_features=64)
        x = torch.zeros(2, 1, 32)
        # Set trigger values
        x[0, 0, 0] = 1.0  # control
        x[1, 0, 10] = 1.0  # treatment
        z = sae.encode(x)
        assert z.shape == (2, 64)
        assert (z[0, :5] > 0.0).all()
        assert (z[1, 10:15] > 0.0).all()

        z, x_hat = sae(x)
        assert x_hat.shape == (2, 32)

    def test_mock_transcoder(self):
        tc = MockTranscoder(source_layer=1, target_layer=2, n_source_features=64, n_target_features=64)
        src = torch.zeros(2, 64)
        tgt = tc(src)
        assert tgt.shape == (2, 64)


class TestMockInterventionStates:
    def test_clamping_sets_mock_intervention_flag(self):
        os.environ["MOCK_PIPELINE"] = "true"
        sae = MockSAE()
        assert not is_mock_intervention_active()
        with clamp_features(sae, [1, 2], 0.0):
            assert is_mock_intervention_active()
        assert not is_mock_intervention_active()
        del os.environ["MOCK_PIPELINE"]

    def test_steering_sets_mock_intervention_flag(self):
        os.environ["MOCK_PIPELINE"] = "true"
        model = MockModel()
        vector = torch.zeros(128)
        assert not is_mock_intervention_active()
        handle = apply_steering_hook(model, 0, vector, 1.0)
        assert is_mock_intervention_active()
        handle.remove()
        assert not is_mock_intervention_active()
        del os.environ["MOCK_PIPELINE"]

    def test_patching_sets_mock_intervention_flag(self):
        os.environ["MOCK_PIPELINE"] = "true"
        model = MockModel()
        assert not is_mock_intervention_active()
        with patch_activations(model, {}):
            assert is_mock_intervention_active()
        assert not is_mock_intervention_active()
        del os.environ["MOCK_PIPELINE"]
