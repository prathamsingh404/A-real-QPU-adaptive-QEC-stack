"""Tests for online regret analysis module."""

import numpy as np
import pytest

from adaptive_qec.analysis.regret_analysis import (
    compute_adaptive_efficiency,
    compute_cumulative_regret_curve,
    compute_online_regret,
    plot_regret_curve,
)


def test_compute_online_regret_monotonic_and_switching():
    arm_lers = [0.03, 0.03, 0.025, 0.02]
    oracle_lers = [0.02, 0.02, 0.02, 0.018]
    summary = compute_online_regret(arm_lers, oracle_lers, adaptation_tax=0.005)

    assert summary.steps == 4
    assert summary.total_regret >= 0
    assert summary.num_switches == 2  # changed at t=2 and t=3
    assert summary.switching_cost_total == 2 * 0.005
    assert len(summary.regret_curve) == 4


def test_compute_cumulative_regret_curve():
    trajectories = [
        [0.0, 0.01, 0.02, 0.03],
        [0.0, 0.015, 0.025, 0.035],
        [0.0, 0.008, 0.018, 0.028],
    ]
    res = compute_cumulative_regret_curve(trajectories)
    assert "mean" in res
    assert "median" in res
    assert "std" in res
    assert len(res["mean"]) == 4
    assert res["mean"][-1] > res["mean"][0]


def test_compute_adaptive_efficiency():
    # cg achieves half the improvement of oracle
    eff = compute_adaptive_efficiency(cg_lers=0.025, oracle_lers=0.02, static_lers=0.03)
    assert eff["valid_range"] is True
    assert pytest.approx(eff["efficiency"], rel=1e-3) == 0.5
    assert pytest.approx(eff["gross_value"], rel=1e-3) == 0.005
    assert pytest.approx(eff["oracle_headroom"], rel=1e-3) == 0.01


def test_plot_regret_curve(tmp_path):
    out = tmp_path / "regret.png"
    curve = [0.0, 0.01, 0.02, 0.03, 0.05]
    saved = plot_regret_curve(curve, out)
    assert saved.exists()
    assert saved.stat().st_size > 0
