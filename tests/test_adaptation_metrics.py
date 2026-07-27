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
