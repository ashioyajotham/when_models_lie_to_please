"""
Circuit validation via ablation studies and activation patching.

Three validation methods:
  1. Ablation: Clamp candidate override features to zero; measure behavioral restoration.
     If faithfulness/non-sycophancy is restored, the features are causally necessary.

  2. Activation patching: Replace activations in the unfaithful/sycophantic forward pass
     with activations from the faithful/non-sycophantic forward pass at specific layers.
     The layer where patching restores behavior is the divergence point.

  3. Perturbation: Following "Biology of a Large Language Model" methodology —
     inhibit groups of features and measure downstream effects on other features
     and final model outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.gemma_scope import JumpReLUSAE
from src.utils.mishax_wrapper import clamp_features, patch_activations

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    feature_indices: list[int]
    layer: int
    clamp_value: float
    faithfulness_delta: float    # Positive = improvement
    sycophancy_delta: float      # Positive = reduction in sycophancy rate
    accuracy_delta: float        # Positive = improvement in accuracy


@dataclass
class PatchingResult:
    patched_layers: list[int]
    source_condition: str
    target_condition: str
    faithfulness_delta: float
    sycophancy_delta: float
    accuracy_delta: float


class CircuitValidator:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        saes: dict[int, JumpReLUSAE],
        generate_fn: Callable,
        eval_fn: Callable,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.saes = saes
        self.generate_fn = generate_fn
        self.eval_fn = eval_fn
        self.device = device

    def run_ablation(
        self,
        prompts: list[str],
        correct_answers: list[str],
        feature_indices: list[int],
        layer: int,
        clamp_value: float = 0.0,
        baseline_metrics: dict | None = None,
    ) -> AblationResult:
        """
        Clamp specified features to clamp_value and measure behavioral change.
        """
        sae = self.saes[layer]
        with clamp_features(sae, feature_indices, clamp_value):
            outputs = self.generate_fn(self.model, self.tokenizer, prompts)
            metrics = self.eval_fn(outputs, correct_answers)

        if baseline_metrics is None:
            baseline_metrics = {"faithfulness": 0.0, "sycophancy_rate": 1.0, "accuracy": 0.0}

        return AblationResult(
            feature_indices=feature_indices,
            layer=layer,
            clamp_value=clamp_value,
            faithfulness_delta=metrics["faithfulness"] - baseline_metrics["faithfulness"],
            sycophancy_delta=baseline_metrics["sycophancy_rate"] - metrics["sycophancy_rate"],
            accuracy_delta=metrics["accuracy"] - baseline_metrics["accuracy"],
        )

    def run_activation_patching(
        self,
        treatment_prompts: list[str],
        correct_answers: list[str],
        source_activations: dict[int, torch.Tensor],
        layers_to_patch: list[int] | None,
        source_condition: str = "control",
        target_condition: str = "treatment",
    ) -> list[PatchingResult]:
        """
        Patch activations from source (faithful) into target (unfaithful) forward passes
        across different layer combinations, returning one result per patch configuration.

        If layers_to_patch is None, sweeps all layers individually.
        """
        all_layers = sorted(source_activations.keys())
        patch_configs = (
            [[l] for l in all_layers] if layers_to_patch is None else [layers_to_patch]
        )

        results = []
        for layers in patch_configs:
            patches = {
                layer: {"resid_post": source_activations[layer]}
                for layer in layers
                if layer in source_activations
            }
            with patch_activations(self.model, patches):
                outputs = self.generate_fn(self.model, self.tokenizer, treatment_prompts)
                metrics = self.eval_fn(outputs, correct_answers)

            results.append(
                PatchingResult(
                    patched_layers=layers,
                    source_condition=source_condition,
                    target_condition=target_condition,
                    faithfulness_delta=metrics.get("faithfulness_delta", 0.0),
                    sycophancy_delta=metrics.get("sycophancy_delta", 0.0),
                    accuracy_delta=metrics.get("accuracy_delta", 0.0),
                )
            )
            logger.info(
                "Patching layers %s: faithfulness_delta=%.3f, sycophancy_delta=%.3f",
                layers, results[-1].faithfulness_delta, results[-1].sycophancy_delta,
            )

        return results
