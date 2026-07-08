"""
Conditional Activation Steering (CAST).

Applies steering interventions only when a classifier detects that the override
circuit is currently active. This avoids over-correction on prompts where the
model is reasoning faithfully or genuinely (not sycophantically) agreeing.

Architecture:
  1. At each token step, the override classifier observes the residual stream
  2. If P(override active) > threshold, apply steering
  3. Otherwise, let generation proceed unmodified

This is a token-level intervention rather than a prompt-level one. It is more
precise but requires the classifier to operate at inference speed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.interventions.steering import SteeringVector

logger = logging.getLogger(__name__)


@dataclass
class CASTConfig:
    classifier: torch.nn.Module      # Trained override classifier (from Phase 4)
    classifier_layer: int            # Layer to extract features from for classification
    classifier_threshold: float      # P(override) threshold for intervention
    steering_vectors: list[SteeringVector]
    log_intervention_rate: bool = True


class ConditionalSteerer:
    """
    Wraps a model with conditional activation steering.

    At each token step, extracts residual stream from classifier_layer,
    runs the override classifier, and conditionally applies steering.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: CASTConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._n_interventions = 0
        self._n_steps = 0

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 512,
    ) -> list[str]:
        """
        Generate with CAST applied.

        Note: This implementation uses a custom generation loop rather than
        model.generate() to support per-token classification. For production use,
        consider integrating with HuggingFace's LogitsProcessor API.
        """
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(next(self.model.parameters()).device)

        generated_ids = inputs["input_ids"].clone()
        self._n_interventions = 0
        self._n_steps = 0

        for step in range(max_new_tokens):
            override_prob = self._classify_current_state(generated_ids)

            if override_prob > self.config.classifier_threshold:
                handles = self._register_steering_hooks()
                self._n_interventions += 1
            else:
                handles = []

            with torch.no_grad():
                outputs = self.model(input_ids=generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_ids = next_token_logits.argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=1)

            for handle in handles:
                handle.remove()

            self._n_steps += 1

            # Stop if all sequences have generated EOS
            if (next_token_ids == self.tokenizer.eos_token_id).all():
                break

        if self.config.log_intervention_rate and self._n_steps > 0:
            logger.info(
                "CAST: intervened on %d/%d token steps (%.1f%%)",
                self._n_interventions,
                self._n_steps,
                100 * self._n_interventions / self._n_steps,
            )

        input_length = inputs["input_ids"].shape[1]
        return [
            self.tokenizer.decode(ids[input_length:], skip_special_tokens=True)
            for ids in generated_ids
        ]

    def _classify_current_state(self, input_ids: torch.Tensor) -> float:
        """Run classifier on residual stream and return P(override active)."""
        residual = {}
        if hasattr(self.model.model, "language_model"):
            target_layer = self.model.model.language_model.layers[self.config.classifier_layer]
        else:
            target_layer = self.model.model.layers[self.config.classifier_layer]

        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            residual["h"] = h[:, -1, :].detach()

        handle = target_layer.register_forward_hook(hook)
        with torch.no_grad():
            self.model(input_ids=input_ids)
        handle.remove()

        if "h" not in residual:
            return 0.0

        with torch.no_grad():
            logits = self.config.classifier(residual["h"])
            prob = torch.sigmoid(logits).mean().item()
        return float(prob)

    def _register_steering_hooks(self) -> list:
        handles = []
        for sv in self.config.steering_vectors:
            if hasattr(self.model.model, "language_model"):
                layer = self.model.model.language_model.layers[sv.layer]
            else:
                layer = self.model.model.layers[sv.layer]
            v = sv.direction.to(next(self.model.parameters()).device)
            m = sv.multiplier

            def make_hook(vec, mul):
                def hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    h = h + mul * vec.unsqueeze(0).unsqueeze(0)
                    return (h,) + output[1:] if isinstance(output, tuple) else h
                return hook

            handles.append(layer.register_forward_hook(make_hook(v, m)))
        return handles
