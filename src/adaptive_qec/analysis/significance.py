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
        baseline: np.ndarray,
        treatment: np.ndarray,
    ) -> HypothesisTestResult:
        """Welch's t-test for unequal variances.

        H₀: μ_baseline = μ_treatment
        H₁: μ_treatment < μ_baseline (treatment has lower error rate)
        """
        t_stat, p_value = stats.ttest_ind(
            baseline, treatment, equal_var=False, alternative="greater"
        )

        # Cohen's d effect size
        pooled_std = np.sqrt(
            (np.var(baseline, ddof=1) + np.var(treatment, ddof=1)) / 2.0
        )
        d = (float(np.mean(baseline)) - float(np.mean(treatment))) / max(float(pooled_std), 1e-10)

        # CI on the difference
        diff = baseline.mean() - treatment.mean()
        se = np.sqrt(np.var(baseline, ddof=1)/len(baseline) + np.var(treatment, ddof=1)/len(treatment))
        t_crit = stats.t.ppf(1 - self._alpha/2, df=min(len(baseline), len(treatment)) - 1)
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
            sample_sizes=(len(baseline), len(treatment)),
        )

    def mann_whitney_test(
        self,
        baseline: np.ndarray,
        treatment: np.ndarray,
    ) -> HypothesisTestResult:
        """Mann-Whitney U test (non-parametric).
