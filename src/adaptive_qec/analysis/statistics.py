"""
Statistical analysis utilities.

Do not report:
    "Model improved from 2.31% to 2.19%."
and stop.

Report confidence intervals. Run enough shots.
Compare H₀: p_{L,A} = p_{L,B} where appropriate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def compute_confidence_interval(
    n_errors: int,
    n_total: int,
    confidence: float = 0.95,
    method: str = "wilson",
) -> tuple[float, float]:
    """
    Compute confidence interval for a binomial proportion (error rate).

    Args:
        n_errors: number of logical errors.
        n_total: total number of shots.
        confidence: confidence level (e.g., 0.95).
        method: "wilson" (recommended), "clopper_pearson" (exact), or "normal".

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    if n_total == 0:
        return (0.0, 1.0)

    p_hat = n_errors / n_total
    alpha = 1 - confidence

    if method == "wilson":
        # Wilson score interval — recommended for small sample sizes
        z = stats.norm.ppf(1 - alpha / 2)
        denominator = 1 + z ** 2 / n_total
        center = (p_hat + z ** 2 / (2 * n_total)) / denominator
        half_width = (z / denominator) * np.sqrt(
            p_hat * (1 - p_hat) / n_total + z ** 2 / (4 * n_total ** 2)
        )
        lower = max(0.0, center - half_width)
        upper = min(1.0, center + half_width)

    elif method == "clopper_pearson":
        # Exact Clopper-Pearson interval
        if n_errors == 0:
            lower = 0.0
        else:
            lower = stats.beta.ppf(alpha / 2, n_errors, n_total - n_errors + 1)
        if n_errors == n_total:
            upper = 1.0
        else:
            upper = stats.beta.ppf(1 - alpha / 2, n_errors + 1, n_total - n_errors)

    elif method == "normal":
        # Normal approximation — only for large n
        z = stats.norm.ppf(1 - alpha / 2)
        se = np.sqrt(p_hat * (1 - p_hat) / n_total)
        lower = max(0.0, p_hat - z * se)
        upper = min(1.0, p_hat + z * se)

    else:
        raise ValueError(f"Unknown CI method: {method}")

    return (float(lower), float(upper))


@dataclass
class HypothesisTestResult:
    """Result of a two-proportion hypothesis test."""
    test_name: str
    p_value: float
    test_statistic: float
    significant: bool
    confidence: float
    effect_size: float  # difference in error rates
    details: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.details is None:
            self.details = {}


def compare_error_rates(
    n_errors_a: int,
    n_total_a: int,
    n_errors_b: int,
    n_total_b: int,
    confidence: float = 0.95,
    alternative: str = "two-sided",
) -> HypothesisTestResult:
    """
    Test H₀: p_{L,A} = p_{L,B}.

