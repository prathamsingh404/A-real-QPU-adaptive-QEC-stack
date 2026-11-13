"""Unit tests for statistical analysis utilities in adaptive_qec.analysis.statistics."""

import numpy as np
import pytest

from adaptive_qec.analysis.statistics import (
    bootstrap_compare,
    cohen_d_paired,
    compare_error_rates,
    compute_confidence_interval,
    paired_ttest,
    paired_wilcoxon_test,
)


def test_compute_confidence_interval():
    # Wilson interval
    low, high = compute_confidence_interval(5, 100, confidence=0.95, method="wilson")
    assert 0.0 < low < 0.05 < high < 0.15

    # Clopper-Pearson exact interval
    low_cp, high_cp = compute_confidence_interval(0, 100, confidence=0.95, method="clopper_pearson")
    assert low_cp == 0.0
    assert 0.0 < high_cp < 0.05

    # Normal approximation
    low_n, high_n = compute_confidence_interval(50, 1000, confidence=0.95, method="normal")
    assert 0.03 < low_n < 0.05 < high_n < 0.07

    # Zero total shots
    assert compute_confidence_interval(0, 0) == (0.0, 1.0)

    # Unknown method
    with pytest.raises(ValueError, match="Unknown CI method"):
        compute_confidence_interval(1, 10, method="invalid")


def test_compare_error_rates():
    # Significant difference
    res = compare_error_rates(10, 1000, 50, 1000, confidence=0.95)
    assert res.significant is True
    assert res.p_value < 0.001
    assert res.effect_size < 0.0

    # No difference
    res_null = compare_error_rates(20, 1000, 20, 1000, confidence=0.95)
    assert res_null.significant is False
    assert res_null.test_statistic == 0.0
    assert res_null.p_value == 1.0


def test_paired_ttest_standard():
    # Adaptive systematically better than static
    rng = np.random.default_rng(42)
    static = np.array([0.15, 0.16, 0.14, 0.15, 0.17, 0.16, 0.15, 0.18])
    adaptive = static - 0.03 + rng.normal(0, 0.002, size=len(static))

    t_stat, p_val = paired_ttest(adaptive, static, alternative="less")
    assert t_stat < -5.0
    assert p_val < 0.001

    # Pass differences directly
    diff = adaptive - static
    t_stat_d, p_val_d = paired_ttest(diff, alternative="less")
    assert np.isclose(t_stat, t_stat_d)
    assert np.isclose(p_val, p_val_d)


def test_paired_ttest_zero_variance():
    # Identical values between static and adaptive (e.g. static and CG-CCADS have identical errors)
    static = np.array([0.05, 0.08, 0.06, 0.04, 0.07])
    adaptive = np.array([0.05, 0.08, 0.06, 0.04, 0.07])

    t_stat, p_val = paired_ttest(adaptive, static)
    assert t_stat == 0.0
    assert p_val == 1.0

    # Constant non-zero difference (zero variance of difference)
    t_stat_const, p_val_const = paired_ttest([0.1, 0.1, 0.1], [0.05, 0.05, 0.05])
    assert t_stat_const == 0.0
    assert p_val_const == 1.0

    # Fewer than 2 pairs
    assert paired_ttest([0.05], [0.05]) == (0.0, 1.0)

    # Length mismatch
    with pytest.raises(ValueError, match="Sample lengths must match"):
        paired_ttest([0.1, 0.2], [0.1])


def test_paired_wilcoxon_test():
    # Non-parametric test with clear separation
    static = np.array([0.20, 0.19, 0.22, 0.25, 0.18, 0.21, 0.24, 0.23, 0.19, 0.20])
    adaptive = static - np.array([0.02, 0.03, 0.01, 0.04, 0.02, 0.03, 0.02, 0.01, 0.02, 0.03])

    stat, pval = paired_wilcoxon_test(adaptive, static, alternative="two-sided", zero_method="pratt")
    assert stat >= 0.0
    assert pval < 0.01

    # Test with zero_method="wilcox"
    stat_w, pval_w = paired_wilcoxon_test(adaptive, static, alternative="two-sided", zero_method="wilcox")
    assert stat_w >= 0.0
    assert pval_w < 0.01

    # Identical errors (zero variance / all zero deltas)
    stat_null, pval_null = paired_wilcoxon_test([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
    assert stat_null == 0.0
    assert pval_null == 1.0

    # Short sample
    assert paired_wilcoxon_test([0.1], [0.1]) == (0.0, 1.0)


def test_cohen_d_paired():
    # Significant negative effect size (adaptive lower LER)
    diffs = np.array([-0.02, -0.025, -0.018, -0.022, -0.021])
    d = cohen_d_paired(diffs)
    expected_d = np.mean(diffs) / np.std(diffs, ddof=1)
    assert np.isclose(d, expected_d)
    assert d < -5.0

    # Two samples passed
    static = np.array([0.10, 0.12, 0.11, 0.13])
    adaptive = np.array([0.08, 0.09, 0.09, 0.10])
    d_two = cohen_d_paired(adaptive, static)
    assert d_two < 0.0

    # Zero variance returns 0.0
    assert cohen_d_paired([0.05, 0.05, 0.05], [0.05, 0.05, 0.05]) == 0.0
    assert cohen_d_paired([0.0, 0.0, 0.0]) == 0.0
    assert cohen_d_paired([0.05]) == 0.0


def test_bootstrap_compare():
    rng = np.random.default_rng(123)
    outcomes_a = rng.binomial(1, 0.10, size=500)
    outcomes_b = rng.binomial(1, 0.25, size=500)
    res = bootstrap_compare(outcomes_a, outcomes_b, n_bootstrap=1000)
    assert res["observed_difference"] < -0.10
    assert res["bootstrap_ci"][0] < res["bootstrap_ci"][1] < 0.0
    assert res["significant"] is True
