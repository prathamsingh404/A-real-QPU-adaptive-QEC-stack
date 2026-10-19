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
