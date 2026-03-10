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
