"""Unit tests for statistical significance tests and multiple testing corrections."""

import numpy as np
import pytest

from adaptive_qec.analysis.significance import (
    ComparisonReport,
    HypothesisTestResult,
    SignificanceTester,
    benjamini_hochberg,
    benjamini_hochberg_correction,
    holm_bonferroni_correction,
    welch_t_test,
    wilson_score_ci,
)


def test_wilson_score_ci():
    low, high = wilson_score_ci(10, 100)
    assert 0.05 < low < 0.10 < high < 0.18

    # Edge cases
    assert wilson_score_ci(0, 0) == (0.0, 1.0)
    low_0, high_0 = wilson_score_ci(0, 100)
    assert low_0 == 0.0
    assert 0.0 < high_0 < 0.05


def test_welch_t_test_and_mann_whitney():
    tester = SignificanceTester(alpha=0.05)
    rng = np.random.default_rng(42)

    # Baseline higher errors (worse) than treatment
    baseline_errors = rng.normal(0.20, 0.02, size=50)
    treatment_errors = rng.normal(0.15, 0.02, size=50)

    res_t = tester.welch_t_test(baseline_errors, treatment_errors)
    assert res_t.significant is True
    assert res_t.p_value < 1e-4
    assert res_t.effect_size > 1.5

    res_mw = tester.mann_whitney_test(baseline_errors, treatment_errors)
    assert res_mw.significant is True
    assert res_mw.p_value < 1e-4

    ci = tester.bootstrap_ci(baseline_errors, treatment_errors)
    assert ci[0] > 0.03  # Difference strictly positive

    # Test full comparison report
    report = tester.compare(
        baseline_rewards=1.0 - baseline_errors,
        treatment_rewards=1.0 - treatment_errors,
    )
    assert "SIGNIFICANT" in report.conclusion
    assert report.absolute_improvement > 0.03


def test_holm_bonferroni_correction_step_down():
    # m = 4, alpha = 0.05
    # p = [0.01, 0.04, 0.02, 0.10]
    # Sorted:
    # k=1: p=0.01 <= 0.05 / 4 = 0.0125 -> reject (orig index 0)
    # k=2: p=0.02 <= 0.05 / 3 = 0.0167 -> FALSE (0.02 > 0.0167) -> stop!
    # Expected: only index 0 rejected
    p_vals = [0.01, 0.04, 0.02, 0.10]
    rej = holm_bonferroni_correction(p_vals, alpha=0.05)
    assert rej == [True, False, False, False]

    # All significant
    # k=1: 0.001 <= 0.05/3 = 0.0167 -> reject
    # k=2: 0.005 <= 0.05/2 = 0.025  -> reject
    # k=3: 0.020 <= 0.05/1 = 0.05   -> reject
    p_all = [0.001, 0.005, 0.020]
    assert holm_bonferroni_correction(p_all, alpha=0.05) == [True, True, True]

    # None significant
    p_none = [0.06, 0.08, 0.20]
    assert holm_bonferroni_correction(p_none, alpha=0.05) == [False, False, False]

    # Empty
    assert holm_bonferroni_correction([]) == []

    # SignificanceTester static method
    assert SignificanceTester.holm_bonferroni(p_vals, alpha=0.05) == [True, False, False, False]


def test_benjamini_hochberg_correction_fdr():
    # Same p-values: [0.01, 0.04, 0.02, 0.10]
    # Under BH FDR at q = 0.05:
    # Sorted:
    # k=1: p=0.01 <= (1/4)*0.05 = 0.0125 (True)
    # k=2: p=0.02 <= (2/4)*0.05 = 0.025  (True)
    # k=3: p=0.04 <= (3/4)*0.05 = 0.0375 (False)
    # k=4: p=0.10 <= (4/4)*0.05 = 0.05   (False)
    # max_k = 2 -> rejects indices 0 and 2!
    p_vals = [0.01, 0.04, 0.02, 0.10]
    rej = benjamini_hochberg_correction(p_vals, q=0.05)
    assert rej == [True, False, True, False]

    # Alias alpha supported
    rej_alpha = benjamini_hochberg_correction(p_vals, alpha=0.05)
    assert rej_alpha == [True, False, True, False]

    # Backward-compatible function alias
    assert benjamini_hochberg(p_vals, alpha=0.05) == [True, False, True, False]
    assert SignificanceTester.benjamini_hochberg(p_vals, alpha=0.05) == [True, False, True, False]

    # All rejected
    p_low = [0.001, 0.002, 0.003]
    assert benjamini_hochberg_correction(p_low, q=0.05) == [True, True, True]

    # None rejected
    p_high = [0.5, 0.6, 0.7]
    assert benjamini_hochberg_correction(p_high, q=0.05) == [False, False, False]

    # Empty
    assert benjamini_hochberg_correction([]) == []
