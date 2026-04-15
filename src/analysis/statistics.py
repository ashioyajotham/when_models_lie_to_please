"""
Statistical utilities for significance testing, effect size estimation,
and confidence interval computation across experiments.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size between two independent groups."""
    pooled_std = np.sqrt((group1.std() ** 2 + group2.std() ** 2) / 2)
    if pooled_std < 1e-10:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def bootstrap_ci(
    values: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for a statistic.

    Returns (lower, upper) bounds.
    """
    rng = np.random.default_rng(seed)
    bootstrapped = [
        statistic(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ]
    alpha = 1 - confidence
    lower = np.percentile(bootstrapped, 100 * alpha / 2)
    upper = np.percentile(bootstrapped, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Apply Bonferroni correction; return corrected p-values capped at 1.0."""
    return np.minimum(p_values * len(p_values), 1.0)


def welch_t_test(
    group1: np.ndarray,
    group2: np.ndarray,
) -> tuple[float, float]:
    """Two-sided Welch t-test. Returns (t_statistic, p_value)."""
    result = stats.ttest_ind(group1, group2, equal_var=False)
    return float(result.statistic), float(result.pvalue)


def mcnemar_test(
    before_correct: np.ndarray,
    after_correct: np.ndarray,
) -> tuple[float, float]:
    """
    McNemar's test for paired proportions.

    Used to assess whether an intervention significantly changes accuracy
    on the same set of prompts.

    Args:
        before_correct: Boolean array of whether each prompt was answered correctly before
        after_correct: Boolean array of whether each prompt was answered correctly after

    Returns:
        (statistic, p_value)
    """
    b = int(np.sum(before_correct & ~after_correct))  # Correct before, wrong after
    c = int(np.sum(~before_correct & after_correct))  # Wrong before, correct after
    if b + c == 0:
        return 0.0, 1.0
    result = stats.mcnemar([[0, b], [c, 0]], correction=True)
    return float(result.statistic), float(result.pvalue)
