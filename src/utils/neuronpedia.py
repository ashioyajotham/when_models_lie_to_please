"""
Neuronpedia API integration for feature autointerpretation and inspection.

Neuronpedia provides:
  - Feature max-activating examples (top-k inputs that fire a given feature)
  - Autointerpretation scores and descriptions
  - Circuit exploration UI (not accessible programmatically, but links are generated)

API documentation: https://www.neuronpedia.org/api-doc
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

NEURONPEDIA_BASE_URL = "https://www.neuronpedia.org/api"
DEFAULT_TIMEOUT = 30


@dataclass
class FeatureInterpretation:
    feature_index: int
    layer: int
    model: str
    description: str
    autointerp_score: float | None
    top_activating_tokens: list[str]
    max_activation: float


class NeuronpediaClient:
    """
    Client for the Neuronpedia public API.

    Requires NEURONPEDIA_API_KEY environment variable.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("NEURONPEDIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Neuronpedia API key required. Set NEURONPEDIA_API_KEY environment variable."
            )
        self.session = requests.Session()
        self.session.headers.update({"x-api-key": self.api_key})

    def get_feature(
        self,
        model_id: str,
        sae_id: str,
        feature_index: int,
    ) -> dict[str, Any]:
        """
        Fetch feature metadata from Neuronpedia.

        Args:
            model_id: Neuronpedia model identifier (e.g., "gemma-3-4b-it")
            sae_id: SAE identifier on Neuronpedia (e.g., "4-llamascope-res-131k")
            feature_index: Index of the feature to fetch
        """
        url = f"{NEURONPEDIA_BASE_URL}/feature/{model_id}/{sae_id}/{feature_index}"
        response = self.session.get(url, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        return response.json()

    def get_features_batch(
        self,
        model_id: str,
        sae_id: str,
        feature_indices: list[int],
        sleep_between_requests: float = 0.1,
    ) -> dict[int, dict[str, Any]]:
        """
        Fetch multiple features, respecting rate limits.
        """
        results = {}
        for idx in feature_indices:
            try:
                results[idx] = self.get_feature(model_id, sae_id, idx)
                time.sleep(sleep_between_requests)
            except requests.HTTPError as exc:
                logger.warning("Failed to fetch feature %d: %s", idx, exc)
        return results

    def autointerp_feature(
        self,
        model_id: str,
        sae_id: str,
        feature_index: int,
    ) -> FeatureInterpretation:
        """
        Retrieve the autointerpretation for a feature and parse into a structured format.
        """
        data = self.get_feature(model_id, sae_id, feature_index)
        return FeatureInterpretation(
            feature_index=feature_index,
            layer=data.get("layer", -1),
            model=model_id,
            description=data.get("explanations", [{}])[0].get("description", ""),
            autointerp_score=data.get("explanations", [{}])[0].get("score"),
            top_activating_tokens=[
                act.get("token", "") for act in data.get("activations", [])[:20]
            ],
            max_activation=data.get("maxActApprox", 0.0),
        )

    def get_circuit_url(self, model_id: str, sae_id: str, feature_index: int) -> str:
        return f"https://www.neuronpedia.org/{model_id}/{sae_id}/{feature_index}"
