"""
Statistical significance testing for adaptive-vs-static comparisons.

Implements hypothesis tests to rigorously determine whether the
adaptive controller's improvement over the static baseline is
statistically significant, not just due to random variation.

Tests:
    1. Two-sample t-test (Welch's) — error rate comparison.
    2. Mann-Whitney U test — non-parametric alternative.
    3. Bootstrap confidence interval — difference in means.
    4. Benjamini-Hochberg correction — multiple comparisons.

All p-values and confidence intervals are reported with full
methodology documentation for paper reproducibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class HypothesisTestResult:
    """Result of a single hypothesis test."""
    test_name: str
    null_hypothesis: str
    alternative_hypothesis: str
    test_statistic: float
    p_value: float
    significant: bool
    alpha: float
    effect_size: float
    confidence_interval: tuple[float, float]
    sample_sizes: tuple[int, int]
    notes: str = ""

    @property
    def statistic(self) -> float:
        """Alias for test_statistic."""
        return self.test_statistic

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "null_hypothesis": self.null_hypothesis,
            "alternative_hypothesis": self.alternative_hypothesis,
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "significant": self.significant,
            "alpha": self.alpha,
            "effect_size": self.effect_size,
            "confidence_interval": list(self.confidence_interval),
            "sample_sizes": list(self.sample_sizes),
            "notes": self.notes,
        }


# Backwards compatibility alias
StatisticalTestResult = HypothesisTestResult


def wilson_score_ci(n_errors: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Accurate and robust for small sample sizes and extreme proportions.
    """
    if n_total == 0:
        return (0.0, 1.0)
    p = n_errors / n_total
    denom = 1.0 + (z ** 2) / n_total
    center = (p + (z ** 2) / (2.0 * n_total)) / denom
    variance_term = (p * (1.0 - p) / n_total) + ((z ** 2) / (4.0 * (n_total ** 2)))
    spread = z * np.sqrt(max(0.0, variance_term)) / denom
    return (max(0.0, float(center - spread)), min(1.0, float(center + spread)))


def welch_t_test(
    baseline: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Standalone Welch's t-test between two arrays of error observations."""
    tester = SignificanceTester(alpha=alpha)
    return tester.welch_t_test(baseline, treatment)


@dataclass
class ComparisonReport:
    """Complete statistical comparison between two controllers."""
    baseline_name: str
    treatment_name: str
    baseline_mean: float
    treatment_mean: float
    absolute_improvement: float
    relative_improvement: float
    tests: list[HypothesisTestResult]
    bootstrap_ci: tuple[float, float]
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_name": self.baseline_name,
            "treatment_name": self.treatment_name,
            "baseline_mean": self.baseline_mean,
            "treatment_mean": self.treatment_mean,
            "absolute_improvement": self.absolute_improvement,
            "relative_improvement": self.relative_improvement,
            "tests": [t.to_dict() for t in self.tests],
            "bootstrap_ci": list(self.bootstrap_ci),
            "conclusion": self.conclusion,
        }


class SignificanceTester:
    """Statistical testing engine for controller comparisons.

    Parameters
    ----------
    alpha : float
        Significance level (default 0.05).
    bootstrap_samples : int
        Number of bootstrap resamples for CI estimation.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        bootstrap_samples: int = 10000,
    ) -> None:
        self._alpha = alpha
        self._bootstrap_n = bootstrap_samples
        self._rng = np.random.default_rng(42)

    def welch_t_test(
        self,
        baseline: Any,
        treatment: Any,
    ) -> HypothesisTestResult:
        """Welch's t-test for unequal variances.

        H₀: μ_baseline = μ_treatment
        H₁: μ_treatment < μ_baseline (treatment has lower error rate)
        """
        b = np.asarray(baseline, dtype=np.float64)
        t = np.asarray(treatment, dtype=np.float64)

        t_stat, p_value = stats.ttest_ind(
            b, t, equal_var=False, alternative="greater"
        )

        # Cohen's d effect size
        pooled_std = np.sqrt(
            (np.var(b, ddof=1) + np.var(t, ddof=1)) / 2.0
        )
        d = (float(np.mean(b)) - float(np.mean(t))) / max(float(pooled_std), 1e-10)

        # CI on the difference
        diff = b.mean() - t.mean()
        se = np.sqrt(np.var(b, ddof=1)/len(b) + np.var(t, ddof=1)/len(t))
        t_crit = stats.t.ppf(1 - self._alpha/2, df=max(min(len(b), len(t)) - 1, 1))
        ci = (float(diff - t_crit * se), float(diff + t_crit * se))

        return HypothesisTestResult(
            test_name="Welch's t-test",
            null_hypothesis="μ_baseline = μ_treatment",
            alternative_hypothesis="μ_treatment < μ_baseline (lower error rate)",
            test_statistic=float(t_stat),
            p_value=float(p_value),
            significant=float(p_value) < self._alpha,
            alpha=self._alpha,
            effect_size=float(d),
            confidence_interval=ci,
            sample_sizes=(len(b), len(t)),
        )

    def mann_whitney_test(
        self,
        baseline: Any,
        treatment: Any,
    ) -> HypothesisTestResult:
        """Mann-Whitney U test (non-parametric).

        Does not assume normality. Tests whether treatment values
        tend to be smaller than baseline values.
        """
        b = np.asarray(baseline, dtype=np.float64)
        t = np.asarray(treatment, dtype=np.float64)

        u_stat, p_value = stats.mannwhitneyu(
            b, t, alternative="greater"
        )

        # Rank-biserial correlation as effect size
        n1, n2 = len(b), len(t)
        r = 1 - 2 * u_stat / (n1 * n2)

        return HypothesisTestResult(
            test_name="Mann-Whitney U test",
            null_hypothesis="P(baseline > treatment) = 0.5",
            alternative_hypothesis="P(baseline > treatment) > 0.5",
            test_statistic=float(u_stat),
            p_value=float(p_value),
            significant=float(p_value) < self._alpha,
