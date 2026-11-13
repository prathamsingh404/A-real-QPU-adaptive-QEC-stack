"""Tests for online regret analysis module and regret analyzer."""

import numpy as np
import pytest

from adaptive_qec.analysis.regret import RegretAnalyzer, RegretAnalysis
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


def test_regret_analyzer_basic_accumulation():
    """Verify cumulative regret and time-averaged regret on a deterministic trajectory."""
    results = [
        {
            "controller_name": "agent_a",
            "action": {"decoder": "mwpm", "dd_policy": "none"},
            "reward": 0.8,
        }
        for _ in range(10)
    ]
    results.append({
        "controller_name": "oracle_agent",
        "action": {"decoder": "uf", "dd_policy": "none"},
        "reward": 1.0,
    })

    analyzer = RegretAnalyzer(results, switch_penalty=0.01)
    analysis = analyzer.analyze("agent_a")

    assert analysis.controller_name == "agent_a"
    assert analysis.total_steps == 10
    assert analysis.total_switches == 0
    assert analysis.switch_regret == 0.0
    assert analysis.controller_total_reward == pytest.approx(8.0)
    assert analysis.controller_mean_reward == pytest.approx(0.8)
    assert len(analysis.regret_curve) == 10


def test_regret_analyzer_switching_penalty():
    """Verify that switching between decoders correctly increments switches and penalty."""
    results = []
    decoders = ["mwpm", "uf", "mwpm", "uf", "mwpm"]  # 4 switches
    for dec in decoders:
        results.append({
            "controller_name": "switcher",
            "action": {"decoder": dec, "dd_policy": "xy4"},
            "reward": 0.9,
        })

    analyzer = RegretAnalyzer(results, switch_penalty=0.05)
    analysis = analyzer.analyze("switcher")

    assert analysis.total_switches == 4
    assert analysis.switch_regret == pytest.approx(4 * 0.05)


def test_regret_analyzer_multi_controller_comparison():
    """Verify comparing multiple controllers and finding the empirical oracle arm."""
    results = []
    for _ in range(20):
        results.append({
            "controller_name": "static_mwpm",
            "action": {"decoder": "mwpm", "dd_policy": "none"},
            "reward": 0.7,
        })
    for _ in range(20):
        results.append({
            "controller_name": "adaptive_ccads",
            "action": {"decoder": "uf", "dd_policy": "cpmg"},
            "reward": 0.95,
        })

    analyzer = RegretAnalyzer(results, switch_penalty=0.002)
    comparison = analyzer.compare()

    assert "static_mwpm" in comparison
    assert "adaptive_ccads" in comparison

    analysis_static = comparison["static_mwpm"]
    analysis_adapt = comparison["adaptive_ccads"]

    assert analysis_adapt.controller_mean_reward > analysis_static.controller_mean_reward
    assert analysis_static.cumulative_regret > analysis_adapt.cumulative_regret
    assert analysis_adapt.oracle_arm == "uf:cpmg"


def test_regret_analyzer_summary_table_format():
    """Verify that summary_table renders headers, sorted controllers, and oracle arm."""
    analyzer = RegretAnalyzer(switch_penalty=0.01)
    analyzer.add_result({
        "controller_name": "arm1",
        "action": {"decoder": "mwpm", "dd_policy": "none"},
        "reward": 0.5,
    })
    analyzer.add_result({
        "controller_name": "arm2",
        "action": {"decoder": "uf", "dd_policy": "none"},
        "reward": 0.9,
    })

    table = analyzer.summary_table()
    assert "Controller" in table
    assert "Avg Reward" in table
    assert "Cum. Regret" in table
    assert "arm1" in table
    assert "arm2" in table
    assert "Oracle arm:" in table


def test_regret_analyzer_missing_controller_raises():
    """Verify analyze raises ValueError when queried controller was not recorded."""
    analyzer = RegretAnalyzer()
    analyzer.add_result({
        "controller_name": "c1",
        "action": {"decoder": "mwpm", "dd_policy": "none"},
        "reward": 0.5,
    })
    with pytest.raises(ValueError, match="No results for controller: nonexistent"):
        analyzer.analyze("nonexistent")


def test_regret_analysis_serialization():
    """Verify to_dict returns serializable dictionary with all required metrics."""
    results = [
        {
            "controller_name": "test_ctrl",
            "action": {"decoder": "mwpm", "dd_policy": "none"},
            "reward": 0.75,
        }
    ]
    analyzer = RegretAnalyzer(results)
    analysis = analyzer.analyze("test_ctrl")
    d = analysis.to_dict()

    expected_keys = [
        "controller_name", "total_steps", "cumulative_regret",
        "time_averaged_regret", "normalized_regret", "oracle_arm",
        "oracle_total_reward", "controller_total_reward",
        "controller_mean_reward", "total_switches", "switch_regret",
        "reward_std", "reward_p5", "reward_p95",
    ]
    for key in expected_keys:
        assert key in d, f"Missing key {key} in analysis dict"
    assert d["controller_name"] == "test_ctrl"
    assert d["total_steps"] == 1
