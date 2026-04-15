"""
Cross-condition transfer experiments.

Tests whether interventions developed for one failure mode transfer to the other,
providing empirical evidence for (or against) a shared mechanism.

Transfer protocol:
  1. Identify top-k differential features for condition A (e.g., CoT unfaithfulness)
  2. Clamp those features during prompts designed to elicit condition B (e.g., sycophancy)
  3. Measure whether condition B behavior is reduced
  4. Repeat in both directions: A→B and B→A
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.features.differential import DifferentialFeature

logger = logging.getLogger(__name__)


@dataclass
class TransferResult:
    source_condition: str        # Condition whose features were clamped
    target_condition: str        # Condition being tested for transfer
    n_features_clamped: int
    faithfulness_delta: float    # Change in faithfulness score on target_condition prompts
    sycophancy_delta: float      # Change in sycophancy rate on target_condition prompts
    accuracy_delta: float
    transfer_rate: float         # Fraction of effect from single-condition intervention


@dataclass
class TransferSummary:
    cot_to_sycophancy: list[TransferResult]
    sycophancy_to_cot: list[TransferResult]
    shared_mechanism_evidence: str   # "strong", "moderate", "weak", "none"
    interpretation: str


def assess_shared_mechanism(
    cot_to_syco_results: list[TransferResult],
    syco_to_cot_results: list[TransferResult],
    transfer_threshold: float = 0.3,
) -> TransferSummary:
    """
    Assess whether cross-condition transfer results support a shared mechanism hypothesis.

    A transfer rate of >= transfer_threshold in both directions is considered
    "strong" evidence for a shared mechanism.

    Args:
        transfer_threshold: Minimum bidirectional transfer rate for "moderate" evidence.

    Returns:
        TransferSummary with evidence classification.
    """
    # Use the result with the best feature count (n=20) as the primary assessment
    def best_result(results: list[TransferResult]) -> TransferResult | None:
        if not results:
            return None
        return max(results, key=lambda r: r.transfer_rate)

    cot_best = best_result(cot_to_syco_results)
    syco_best = best_result(syco_to_cot_results)

    if cot_best is None or syco_best is None:
        return TransferSummary(
            cot_to_sycophancy=cot_to_syco_results,
            sycophancy_to_cot=syco_to_cot_results,
            shared_mechanism_evidence="none",
            interpretation="Insufficient data for assessment.",
        )

    cot_rate = cot_best.transfer_rate
    syco_rate = syco_best.transfer_rate
    bidirectional = min(cot_rate, syco_rate)

    if bidirectional >= transfer_threshold * 2:
        evidence = "strong"
        interp = (
            f"Bidirectional transfer rate {bidirectional:.2f} exceeds strong threshold. "
            f"Features causally involved in CoT unfaithfulness are also causally involved in "
            f"sycophancy, and vice versa. This supports the shared override circuit hypothesis."
        )
    elif bidirectional >= transfer_threshold:
        evidence = "moderate"
        interp = (
            f"Bidirectional transfer rate {bidirectional:.2f} meets moderate threshold. "
            f"Partial overlap in causal mechanisms. Possible: shared upstream component with "
            f"condition-specific downstream expression."
        )
    elif max(cot_rate, syco_rate) >= transfer_threshold:
        evidence = "weak"
        interp = (
            f"Unidirectional transfer only (CoT→Syco: {cot_rate:.2f}, Syco→CoT: {syco_rate:.2f}). "
            f"Asymmetric overlap may indicate a hierarchical relationship rather than a fully "
            f"shared circuit."
        )
    else:
        evidence = "none"
        interp = (
            f"No meaningful transfer in either direction (CoT→Syco: {cot_rate:.2f}, "
            f"Syco→CoT: {syco_rate:.2f}). CoT unfaithfulness and sycophancy appear to be "
            f"mechanistically independent. Both are valuable negative results."
        )

    logger.info("Shared mechanism assessment: %s", evidence)
    logger.info(interp)

    return TransferSummary(
        cot_to_sycophancy=cot_to_syco_results,
        sycophancy_to_cot=syco_to_cot_results,
        shared_mechanism_evidence=evidence,
        interpretation=interp,
    )
