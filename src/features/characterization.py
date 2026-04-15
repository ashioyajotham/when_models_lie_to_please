"""
Feature characterization via Neuronpedia autointerpretation.

For each top-k differential feature, retrieves:
  - Neuronpedia description and autointerp score
  - Max-activating token examples
  - A manual classification label (if assigned)

Feature classification taxonomy (from CLAUDE.md Section 3.1.3):
  correctness_signal      — "I know the correct answer" / internal confidence
  user_agreement_signal   — "The user believes X" / social calibration
  confidence_modulator    — Scales certainty/hedging in output
  reasoning_step_marker   — Marks a step in the CoT sequence
  override_trigger        — Initiates suppression of correctness signal
  output_formatter        — Shapes output style/format
  unknown                 — Cannot be classified from available evidence
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from src.features.differential import DifferentialFeature
from src.utils.neuronpedia import FeatureInterpretation, NeuronpediaClient

logger = logging.getLogger(__name__)

FeatureLabel = Literal[
    "correctness_signal",
    "user_agreement_signal",
    "confidence_modulator",
    "reasoning_step_marker",
    "override_trigger",
    "output_formatter",
    "unknown",
]


@dataclass
class CharacterizedFeature:
    differential: DifferentialFeature
    interpretation: FeatureInterpretation | None
    label: FeatureLabel
    label_confidence: float   # 0.0–1.0; low for heuristic-assigned labels
    notes: str


class FeatureCharacterizer:
    """
    Runs autointerpretation on top-k differential features and assigns
    classification labels based on description keywords and activation patterns.
    """

    def __init__(
        self,
        neuronpedia_client: NeuronpediaClient,
        model_neuronpedia_id: str,
        sae_neuronpedia_id: str,
    ) -> None:
        self.client = neuronpedia_client
        self.model_id = model_neuronpedia_id
        self.sae_id = sae_neuronpedia_id

    def characterize(
        self,
        features: list[DifferentialFeature],
        top_k: int = 50,
    ) -> list[CharacterizedFeature]:
        """
        Fetch autointerpretation for up to top_k features and assign labels.
        """
        top_features = sorted(features, key=lambda f: abs(f.delta_activation), reverse=True)[:top_k]
        characterized = []
        for feat in top_features:
            interpretation = None
            try:
                interpretation = self.client.autointerp_feature(
                    self.model_id, self.sae_id, feat.feature_index
                )
            except Exception as exc:
                logger.warning(
                    "Autointerpretation failed for feature %d at layer %d: %s",
                    feat.feature_index, feat.layer, exc,
                )
            label, confidence, notes = self._assign_label(feat, interpretation)
            characterized.append(
                CharacterizedFeature(
                    differential=feat,
                    interpretation=interpretation,
                    label=label,
                    label_confidence=confidence,
                    notes=notes,
                )
            )
        return characterized

    def _assign_label(
        self,
        feat: DifferentialFeature,
        interp: FeatureInterpretation | None,
    ) -> tuple[FeatureLabel, float, str]:
        if interp is None:
            return "unknown", 0.0, "No autointerpretation available."

        desc = interp.description.lower()
        delta = feat.delta_activation

        # Heuristic keyword matching — to be refined with manual inspection
        if any(kw in desc for kw in ["correct", "answer", "confident", "sure", "know"]):
            if delta < 0:
                return "correctness_signal", 0.6, "Suppressed in treatment; matches 'correct answer' description."
            return "correctness_signal", 0.5, "Active in treatment; review manually."

        if any(kw in desc for kw in ["agree", "user", "person", "they said", "opinion"]):
            return "user_agreement_signal", 0.6, "Matches user-agreement description."

        if any(kw in desc for kw in ["think", "believe", "probably", "might", "uncertain"]):
            return "confidence_modulator", 0.5, "Matches hedging/certainty description."

        if any(kw in desc for kw in ["step", "first", "then", "because", "therefore"]):
            return "reasoning_step_marker", 0.5, "Matches reasoning chain description."

        if any(kw in desc for kw in ["override", "suppress", "ignore", "instead", "however"]):
            return "override_trigger", 0.7, "Description suggests suppression behavior."

        if any(kw in desc for kw in ["format", "list", "bullet", "paragraph"]):
            return "output_formatter", 0.5, "Matches formatting description."

        return "unknown", 0.3, f"No keyword match. Description: '{interp.description[:100]}'"


def save_characterizations(
    characterized: list[CharacterizedFeature],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for cf in characterized:
        record = {
            "label": cf.label,
            "label_confidence": cf.label_confidence,
            "notes": cf.notes,
            "feature_index": cf.differential.feature_index,
            "layer": cf.differential.layer,
            "delta_activation": cf.differential.delta_activation,
            "effect_size": cf.differential.effect_size,
            "interpretation_description": (
                cf.interpretation.description if cf.interpretation else None
            ),
            "autointerp_score": (
                cf.interpretation.autointerp_score if cf.interpretation else None
            ),
        }
        records.append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
