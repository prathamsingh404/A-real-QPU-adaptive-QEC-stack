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
from typing import Any, Optional, Sequence

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
            alpha=self._alpha,
            effect_size=float(r),
            confidence_interval=(0.0, 0.0),  # not applicable for MWU
            sample_sizes=(n1, n2),
            notes="Rank-biserial correlation used as effect size",
        )

    def bootstrap_ci(
        self,
        baseline: Any,
        treatment: Any,
    ) -> tuple[float, float]:
        """Bootstrap confidence interval for the difference in means.

        Returns the (alpha/2, 1-alpha/2) percentile CI.
        """
        b = np.asarray(baseline, dtype=np.float64)
        t = np.asarray(treatment, dtype=np.float64)
        diffs: list[float] = []
        n_b, n_t = len(b), len(t)

        for _ in range(self._bootstrap_n):
            b_sample = self._rng.choice(b, size=n_b, replace=True)
            t_sample = self._rng.choice(t, size=n_t, replace=True)
            diffs.append(float(np.mean(b_sample) - np.mean(t_sample)))

        lower = float(np.percentile(diffs, 100 * self._alpha / 2))

        upper = float(np.percentile(diffs, 100 * (1 - self._alpha / 2)))
        return (lower, upper)

    def compare(
        self,
        baseline_rewards: np.ndarray,
        treatment_rewards: np.ndarray,
        baseline_name: str = "static",
        treatment_name: str = "adaptive",
    ) -> ComparisonReport:
        """Run full statistical comparison between two controllers.

        Uses error rates (1 - reward) for the comparison since
        lower error rate = better controller.

        Parameters
        ----------
        baseline_rewards : np.ndarray
            Per-window rewards from the baseline controller.
        treatment_rewards : np.ndarray
            Per-window rewards from the treatment controller.

        Returns
        -------
        ComparisonReport
            Complete statistical analysis.
        """
        # Convert to error rates for the comparison
        baseline_errors = 1.0 - baseline_rewards
        treatment_errors = 1.0 - treatment_rewards

        # Run tests
        t_test = self.welch_t_test(baseline_errors, treatment_errors)
        mw_test = self.mann_whitney_test(baseline_errors, treatment_errors)
        boot_ci = self.bootstrap_ci(baseline_errors, treatment_errors)

        baseline_mean = float(np.mean(baseline_errors))
        treatment_mean = float(np.mean(treatment_errors))
        abs_improvement = baseline_mean - treatment_mean
        rel_improvement = abs_improvement / max(baseline_mean, 1e-10)

        # Conclusion
        if t_test.significant and mw_test.significant:
            conclusion = (
                f"SIGNIFICANT: {treatment_name} achieves a "
                f"{rel_improvement*100:.2f}% lower error rate than {baseline_name} "
                f"(Welch p={t_test.p_value:.4f}, MW p={mw_test.p_value:.4f}, "
                f"Cohen's d={t_test.effect_size:.3f})"
            )
        elif t_test.significant or mw_test.significant:
            conclusion = (
                f"MIXED: One test is significant, one is not. "
                f"Welch p={t_test.p_value:.4f}, MW p={mw_test.p_value:.4f}. "
                f"Collect more data."
            )
        else:
            conclusion = (
                f"NOT SIGNIFICANT: Cannot conclude {treatment_name} is better "
                f"than {baseline_name} at α={self._alpha} "
                f"(Welch p={t_test.p_value:.4f}, MW p={mw_test.p_value:.4f})"
            )

        return ComparisonReport(
            baseline_name=baseline_name,
            treatment_name=treatment_name,
            baseline_mean=baseline_mean,
            treatment_mean=treatment_mean,
            absolute_improvement=abs_improvement,
            relative_improvement=rel_improvement,
            tests=[t_test, mw_test],
            bootstrap_ci=boot_ci,
            conclusion=conclusion,
        )

    @staticmethod
    def benjamini_hochberg(
        p_values: Sequence[float],
        alpha: float = 0.05,
    ) -> list[bool]:
        """Benjamini-Hochberg FDR correction for multiple comparisons.

        Returns a boolean mask: True = reject null at controlled FDR.
        """
        return benjamini_hochberg_correction(p_values, q=alpha)

    @staticmethod
    def holm_bonferroni(
        p_values: Sequence[float],
        alpha: float = 0.05,
    ) -> list[bool]:
        """Holm-Bonferroni FWER step-down correction for multiple comparisons.

        Returns a boolean mask: True = reject null at controlled FWER.
        """
        return holm_bonferroni_correction(p_values, alpha=alpha)


def holm_bonferroni_correction(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> list[bool]:
    """
    Holm-Bonferroni step-down multiple testing correction.

    Controls the Family-Wise Error Rate (FWER) at level alpha under arbitrary
    or positive dependence.
    Sorts p-values in ascending order p_(1) <= ... <= p_(m) and applies
    step-down criterion: rejects H_(k) if p_(k) <= alpha / (m - k + 1)
    for all j <= k.

    Args:
        p_values: Sequence of p-values to correct.
        alpha: Family-wise significance level (default 0.05).

    Returns:
        list[bool]: Boolean mask in original input order (True = reject H0).
    """
    m = len(p_values)
    if m == 0:
        return []

    sorted_indices = np.argsort(p_values)
    sorted_p = np.asarray(p_values, dtype=np.float64)[sorted_indices]

    rejected_sorted = [False] * m
    for k_idx, p_val in enumerate(sorted_p):
        k = k_idx + 1  # 1-based rank
        threshold = alpha / (m - k + 1)
        if p_val <= threshold:
            rejected_sorted[k_idx] = True
        else:
            # Step-down stops at first non-rejection
            break

    result = [False] * m
    for orig_pos, sorted_idx in enumerate(sorted_indices):
        result[sorted_idx] = rejected_sorted[orig_pos]

    return result


def benjamini_hochberg_correction(
    p_values: Sequence[float],
    q: float = 0.05,
    alpha: Optional[float] = None,
) -> list[bool]:
    """
    Benjamini-Hochberg False Discovery Rate (FDR) control procedure.

    Controls the expected proportion of false discoveries at level q.
    Sorts p-values p_(1) <= ... <= p_(m) and finds the largest k such that
    p_(k) <= (k / m) * q, rejecting all H_(1) through H_(k).

    Args:
        p_values: Sequence of p-values to correct.
        q: False discovery rate target (default 0.05).
        alpha: Alias for q if provided.

    Returns:
        list[bool]: Boolean mask in original input order (True = reject H0).
    """
    if alpha is not None:
        q = alpha

    m = len(p_values)
    if m == 0:
        return []

    sorted_indices = np.argsort(p_values)
    sorted_p = np.asarray(p_values, dtype=np.float64)[sorted_indices]

    thresholds = (np.arange(1, m + 1) / m) * q
    below = sorted_p <= thresholds

    if not np.any(below):
        return [False] * m

    # Find largest k such that p_(k) <= (k/m)*q
    max_k_idx = int(np.max(np.where(below)[0]))

    rejected_sorted = [False] * m
    for i in range(max_k_idx + 1):
        rejected_sorted[i] = True

    result = [False] * m
    for sorted_rank, orig_idx in enumerate(sorted_indices):
        result[orig_idx] = rejected_sorted[sorted_rank]

    return result

    @staticmethod
    def holm_bonferroni(
        p_values: list[float],
        alpha: float = 0.05,
    ) -> list[bool]:
        """Holm-Bonferroni step-down procedure for family-wise error rate (FWER) control.

        Controls FWER at level alpha under arbitrary dependence. Strictly more powerful
        than standard Bonferroni correction.

        Parameters
        ----------
        p_values : list[float]
            List of unadjusted p-values.
        alpha : float
            Target family-wise significance level (default 0.05).

        Returns
        -------
        list[bool]
            Boolean mask in original input order: True = reject null hypothesis.
        """
        n = len(p_values)
        if n == 0:
            return []

        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]

        # Step-down testing
        rejected_sorted = [False] * n
        for k in range(n):
            threshold = alpha / (n - k)
            if sorted_p[k] <= threshold:
                rejected_sorted[k] = True
            else:
                # Once one fails, all subsequent hypotheses are retained
                break

        # Map back to original order
        result = [False] * n
        for k, orig_idx in enumerate(sorted_indices):
            result[orig_idx] = bool(rejected_sorted[k])

        return result


benjamini_hochberg = SignificanceTester.benjamini_hochberg
holm_bonferroni = SignificanceTester.holm_bonferroni


