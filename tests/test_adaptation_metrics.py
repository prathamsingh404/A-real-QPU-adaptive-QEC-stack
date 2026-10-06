"""
Tests for Phase 9: Theoretical Modeling, Analytical Proofs, and Adaptation Metrics.
"""

import numpy as np
import pytest

from adaptive_qec.analysis.adaptation_metrics import (
    AnalyticalProofsAndBounds,
    ConformalLERPredictor,
    EmpiricalRegretTracker,
    compute_adaptation_efficiency,
    compute_adaptation_values,
)


def test_compute_adaptation_values():
    """Verify gross and net adaptation value calculations."""
    # Scenario 1: Profitable adaptation (gross value exceeds taxes)
    res_prof = compute_adaptation_values(
        static_ler=0.180,
        raw_adapted_ler=0.150,
        switching_cost=0.005,
        latency_cost=0.002,
    )
    assert res_prof.gross_value == pytest.approx(0.030)
    assert res_prof.total_tax == pytest.approx(0.007)
    assert res_prof.net_value == pytest.approx(0.023)
    assert res_prof.adaptation_profitable is True

    # Scenario 2: Unprofitable adaptation (historical 50k style failure)
    res_unprof = compute_adaptation_values(
        static_ler=0.170,
        raw_adapted_ler=0.168,
        switching_cost=0.010,
        latency_cost=0.005,
    )
    assert res_unprof.gross_value == pytest.approx(0.002)
    assert res_unprof.total_tax == pytest.approx(0.015)
    assert res_unprof.net_value == pytest.approx(-0.013)
    assert res_unprof.adaptation_profitable is False


def test_compute_adaptation_efficiency():
    """Verify adaptation efficiency metric eta = G_observable / G_oracle."""
    # Perfect observable policy
    eff_perfect = compute_adaptation_efficiency(
        static_ler=0.20,
        deployable_ler=0.15,
        oracle_ler=0.15,
    )
    assert eff_perfect.efficiency == pytest.approx(1.0)
    assert eff_perfect.is_valid_range is True

    # 50% efficiency
    eff_half = compute_adaptation_efficiency(
        static_ler=0.20,
        deployable_ler=0.175,
        oracle_ler=0.15,
    )
    assert eff_half.efficiency == pytest.approx(0.5)

    # Harmful policy (negative efficiency)
    eff_harm = compute_adaptation_efficiency(
        static_ler=0.20,
        deployable_ler=0.22,
        oracle_ler=0.15,
    )
    assert eff_harm.efficiency < 0.0
    assert eff_harm.is_valid_range is False


def test_empirical_regret_tracker():
    """Verify cumulative regret tracking against oracle with switching penalties."""
    tracker = EmpiricalRegretTracker(switch_cost=0.01)

    # Step 1: Controller matches oracle
    tracker.record_step("arm_A", 0.10, "arm_A", 0.10)
    # Step 2: Controller matches oracle
    tracker.record_step("arm_A", 0.10, "arm_A", 0.10)
    # Step 3: Controller sub-optimal (loss 0.15 vs 0.10)
    tracker.record_step("arm_A", 0.15, "arm_B", 0.10)
    # Step 4: Controller switches to arm_B (incurs switch cost 0.01)
    tracker.record_step("arm_B", 0.10, "arm_B", 0.10)

    regret = tracker.compute_regret()
    assert regret.total_steps == 4
    assert regret.total_switches == 1
    assert regret.switching_penalty_total == pytest.approx(0.01)
    # Instantaneous regrets: Step 1: 0, Step 2: 0, Step 3: 0.05, Step 4: 0.01 (switch cost)
    assert regret.cumulative_regret == pytest.approx(0.06)
    assert regret.time_averaged_regret == pytest.approx(0.015)


def test_conformal_prediction_intervals():
    """Verify split conformal prediction intervals on validation set."""
    predictor = ConformalLERPredictor(alpha=0.10)

    # Calibration set of 20 points
    preds = [0.15 + 0.005 * i for i in range(20)]
    # True values with max absolute error ~0.008
    trues = [p + (-0.008 if i % 2 == 0 else 0.006) for i, p in enumerate(preds)]

    q_hat = predictor.calibrate(preds, trues)
    assert q_hat > 0.0
    assert q_hat <= 0.010

    # Test prediction
    low, high = predictor.predict_interval(0.18)
    assert low < 0.18 < high
    assert high - low == pytest.approx(2 * q_hat)


def test_analytical_proofs_and_bounds():
    """Verify analytical bounds: switching condition, false switch, boundary threshold, regret."""
    # 1. Switching condition (TASK-088)
    should_switch, lcb, thresh = AnalyticalProofsAndBounds.evaluate_switching_condition(
        delta_hat=0.020,
        sigma_delta=0.003,
        c_switch=0.005,
        c_latency=0.002,
        confidence=0.95,
    )
    # z_0.95 = 1.645 -> LCB = 0.020 - 1.645 * 0.003 = 0.015065 > 0.007 -> True
    assert should_switch is True
    assert lcb > thresh

    # 2. False switching probability bound (TASK-089)
    p_false = AnalyticalProofsAndBounds.false_switching_probability_bound(
        sigma_noise=0.01,
        n_samples=500,
        threshold=0.005,
    )
    # Exponent: -(500 * 2.5e-5) / (2 * 1e-4) = -0.0125 / 0.0002 = -62.5 -> p_false << 1e-15
    assert p_false < 1e-10

    # 3. Adaptation boundary threshold (TASK-090)
    a_thresh = AnalyticalProofsAndBounds.adaptation_boundary_threshold(
        sigma_noise=0.02,
        c_switch=0.005,
        c_latency=0.002,
        window_length=400,
        desired_confidence=0.95,
    )
    # se = 0.02 / 20 = 0.001. z = 1.645. a_thresh = 0.007 + 0.001645 = 0.008645
    assert a_thresh == pytest.approx(0.008645, abs=1e-4)

    # 4. Piecewise stationary regret bound (TASK-091)
    bound = AnalyticalProofsAndBounds.piecewise_stationary_regret_bound(
        total_steps_t=1000,
        num_change_points_s=3,
        num_arms_k=4,
        c_switch=0.01,
    )
    assert bound > 0.0
