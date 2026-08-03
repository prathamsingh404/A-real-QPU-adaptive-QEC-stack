"""
Tests for Randomized Block DD Benchmark (tests/test_dd_randomized_block.py).
"""

import pytest
from adaptive_qec.experiments.dd_comparison import run_randomized_block_dd_benchmark


def test_randomized_block_dd_benchmark_runs():
    """Verify that randomized block benchmark executes and returns all 4 sequences."""
    res = run_randomized_block_dd_benchmark(
        idle_time_us=30.0,
        t2_us=35.0,
        shots_per_block=100,
        seed=42,
    )

    assert "none" in res.performances
    assert "cpmg" in res.performances
    assert "xy4" in res.performances
    assert "xy8" in res.performances
    assert len(res.block_order) == 8  # 2 randomized cycles of 4
    assert res.pulse_damage_threshold_us > 0.0

    d = res.to_dict()
    assert d["optimal_sequence"] in ["none", "cpmg", "xy4", "xy8"]
