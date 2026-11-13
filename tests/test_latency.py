"""
Tests for Latency Profiling and Adaptation Feasibility Map (tests/test_latency.py).
"""

import numpy as np
import pytest

from adaptive_qec.analysis.latency import LatencyProfiler, compute_adaptation_feasibility


def test_latency_profiler_statistics():
    """Verify latency percentiles, throughput, and macro classification."""
    profiler = LatencyProfiler()

    for _ in range(100):
        profiler.record(
            t_meas_us=1.0,
            t_decode_us=5.0,
            t_decision_us=0.5,
            t_comm_us=0.5,
        )

    res = profiler.summarize()
    assert res.p50_us == 7.0
    assert res.p90_us == 7.0
    assert res.mean_us == 7.0
    assert res.throughput_shots_per_s > 100000
    assert res.timescale_classification == "microsecond_feedforward"


def test_adaptation_feasibility_regimes():
    """Verify that slow drift (rho << 1) yields ADAPT while fast jitter (rho >> 1) yields STAY."""
    # Slow drift: tau_drift = 100s, tau_adapt = 2s -> rho = 0.02 << 1
    slow_point = compute_adaptation_feasibility(tau_drift_s=100.0, tau_adapt_s=2.0)
    assert slow_point.timescale_ratio_rho == 0.02
    assert slow_point.is_feasible
    assert slow_point.recommendation == "ADAPT"

    # Fast noise jitter: tau_drift = 1s, tau_adapt = 2s -> rho = 2.0 >> 1
    fast_point = compute_adaptation_feasibility(tau_drift_s=1.0, tau_adapt_s=2.0)
    assert fast_point.timescale_ratio_rho == 2.0
    assert not fast_point.is_feasible
    assert fast_point.recommendation == "STAY"


def test_latency_profiler_empty_fallback():
    """Verify empty profiler returns unprofiled classification with zeroed stats."""
    profiler = LatencyProfiler()
    res = profiler.summarize()
    assert res.timescale_classification == "unprofiled"
    assert res.mean_us == 0.0
    assert res.p50_us == 0.0
    assert res.breakdown_mean.total_us == 0.0


def test_latency_breakdown_and_profile_serialization():
    """Verify serialization to dictionary for both breakdown and profile result."""
    profiler = LatencyProfiler()
    profiler.record(t_meas_us=1.5, t_decode_us=4.5, t_decision_us=1.0, t_comm_us=1.0)
    res = profiler.summarize()

    d_res = res.to_dict()
    assert "p50_us" in d_res
    assert "breakdown_mean" in d_res
    assert d_res["breakdown_mean"]["t_meas_us"] == 1.5
    assert d_res["breakdown_mean"]["total_us"] == 8.0
    assert d_res["timescale_classification"] == "microsecond_feedforward"


def test_latency_profiler_macro_orchestration_classification():
    """Verify latency >= 10 us triggers macro_orchestration timescale classification."""
    profiler = LatencyProfiler()
    profiler.record(t_meas_us=5.0, t_decode_us=8.0, t_decision_us=2.0, t_comm_us=1.0)
    res = profiler.summarize()
    assert res.mean_us == 16.0
    assert res.timescale_classification == "macro_orchestration"


def test_adaptation_feasibility_critical_boundary():
    """Verify behavior near the rho=0.3 critical boundary."""
    # Just below boundary: rho = 0.28 < 0.3
    pt_feasible = compute_adaptation_feasibility(tau_drift_s=10.0, tau_adapt_s=2.8, base_benefit=0.08)
    assert pt_feasible.timescale_ratio_rho == pytest.approx(0.28, rel=1e-3)
    assert pt_feasible.is_feasible
    assert pt_feasible.recommendation == "ADAPT"

    # Just above boundary: rho = 0.32 > 0.3
    pt_infeasible = compute_adaptation_feasibility(tau_drift_s=10.0, tau_adapt_s=3.2, base_benefit=0.08)
    assert pt_infeasible.timescale_ratio_rho == pytest.approx(0.32, rel=1e-3)
    assert not pt_infeasible.is_feasible
    assert pt_infeasible.recommendation == "STAY"

