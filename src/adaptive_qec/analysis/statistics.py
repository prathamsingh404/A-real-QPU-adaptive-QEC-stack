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

    Uses a two-proportion z-test.

    Args:
        n_errors_a, n_total_a: errors and shots for decoder/config A.
        n_errors_b, n_total_b: errors and shots for decoder/config B.
        confidence: confidence level.
        alternative: "two-sided", "less", or "greater".

    Returns:
        HypothesisTestResult with p-value and significance.
    """
    alpha = 1 - confidence

    if np.isnan(n_errors_a) or np.isnan(n_errors_b) or np.isnan(n_total_a) or np.isnan(n_total_b):
        return HypothesisTestResult(
            test_name="two_proportion_z_test",
            p_value=1.0,
            test_statistic=0.0,
            significant=False,
            confidence=confidence,
            effect_size=0.0,
            details={"error": "NaN encountered in inputs"},
        )

    p_a = n_errors_a / n_total_a if n_total_a > 0 else 0
    p_b = n_errors_b / n_total_b if n_total_b > 0 else 0

    # Pooled proportion under H₀
    p_pool = (n_errors_a + n_errors_b) / (n_total_a + n_total_b)

    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_total_a + 1 / n_total_b))

    if se < 1e-12:
        z_stat = 0.0
    else:
        z_stat = (p_a - p_b) / se

    # P-value
    if alternative == "two-sided":
        p_value = 2 * stats.norm.sf(abs(z_stat))
    elif alternative == "less":
        p_value = stats.norm.cdf(z_stat)
    elif alternative == "greater":
        p_value = stats.norm.sf(z_stat)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    significant = bool(p_value < alpha)
    effect_size = p_a - p_b

    result = HypothesisTestResult(
        test_name="two_proportion_z_test",
        p_value=float(p_value),
        test_statistic=float(z_stat),
        significant=significant,
        confidence=confidence,
        effect_size=float(effect_size),
        details={
            "p_a": float(p_a),
            "p_b": float(p_b),
            "n_a": n_total_a,
            "n_b": n_total_b,
            "pooled_p": float(p_pool),
            "alternative": alternative,
        },
    )

    logger.info(
        f"Hypothesis test: p_A={p_a:.6f} vs p_B={p_b:.6f}, "
        f"z={z_stat:.3f}, p={p_value:.6f}, "
        f"{'SIGNIFICANT' if significant else 'not significant'}"
    )
    return result


def bootstrap_compare(
    outcomes_a: np.ndarray,
    outcomes_b: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> dict:
    """
    Bootstrap comparison of two decoder/experiment outcomes.

    Computes the distribution of the difference in error rates
    via resampling.

    Args:
        outcomes_a: binary array (shots_a,) — 1 = error, 0 = correct
        outcomes_b: binary array (shots_b,) — 1 = error, 0 = correct
        n_bootstrap: number of bootstrap samples
        confidence: confidence level

    Returns:
        Dict with bootstrap CI, p-value, and effect size.
    """
    a = np.asarray(outcomes_a, dtype=np.float64)
    b = np.asarray(outcomes_b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return {
            "observed_difference": 0.0,
            "bootstrap_ci": [0.0, 0.0],
            "bootstrap_p_value": 1.0,
            "significant": False,
            "n_bootstrap": n_bootstrap,
        }

    rng = np.random.default_rng(42)

    diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample_a = rng.choice(a, size=len(a), replace=True)
        sample_b = rng.choice(b, size=len(b), replace=True)
        diffs[i] = sample_a.mean() - sample_b.mean()

    alpha = 1 - confidence
    ci_low = float(np.percentile(diffs, 100 * alpha / 2))
    ci_high = float(np.percentile(diffs, 100 * (1 - alpha / 2)))

    # Bootstrap p-value (two-sided)
    observed_diff = a.mean() - b.mean()
    p_value = float(np.mean(np.abs(diffs) >= abs(observed_diff)))

    return {
        "observed_difference": float(observed_diff),
        "bootstrap_ci": [ci_low, ci_high],
        "bootstrap_p_value": p_value,
        "significant": p_value < alpha,
        "n_bootstrap": n_bootstrap,
    }


def wilcoxon_signed_rank_test(
    outcomes_a: np.ndarray,
    outcomes_b: np.ndarray,
    confidence: float = 0.95,
    alternative: str = "two-sided",
) -> HypothesisTestResult:
    """Wilcoxon signed-rank test for paired observations with NaN filtering.

    Args:
        outcomes_a: Array of paired observations from method A.
        outcomes_b: Array of paired observations from method B.
        confidence: Confidence level.
        alternative: "two-sided", "greater", or "less".

    Returns:
        HypothesisTestResult with Wilcoxon statistic and p-value.
    """
    a = np.asarray(outcomes_a, dtype=np.float64)
    b = np.asarray(outcomes_b, dtype=np.float64)

    # Filter NaNs and Infs
    mask = np.isfinite(a) & np.isfinite(b)
    a_clean = a[mask]
    b_clean = b[mask]

    diff = a_clean - b_clean
    nonzero_diff = diff[diff != 0.0]

    alpha = 1.0 - confidence
    if len(nonzero_diff) == 0:
        return HypothesisTestResult(
            test_name="wilcoxon_signed_rank_test",
            p_value=1.0,
            test_statistic=0.0,
            significant=False,
            confidence=confidence,
            effect_size=0.0,
            details={"n_pairs": len(a_clean), "n_nonzero_diff": 0, "alternative": alternative},
        )

    res = stats.wilcoxon(diff, alternative=alternative)
    stat = float(res.statistic)
    p_val = float(res.pvalue)

    return HypothesisTestResult(
        test_name="wilcoxon_signed_rank_test",
        p_value=p_val,
        test_statistic=stat,
        significant=bool(p_val < alpha),
        confidence=confidence,
        effect_size=float(np.median(diff)),
        details={"n_pairs": len(a_clean), "n_nonzero_diff": len(nonzero_diff), "alternative": alternative},
    )

