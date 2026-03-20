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
