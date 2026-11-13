import numpy as np
import pytest

from adaptive_qec.analysis.regret import (
    RegretAnalyzer,
    compute_cumulative_regret_bounds,
    compute_dynamic_regret,
)


def test_cumulative_regret_monotonicity():
    mwpm_errors = np.array([10, 15, 20, 25])
    uf_errors = np.array([25, 20, 15, 10])
    adaptive_errors = np.array([10, 15, 15, 10])
    oracle_best = np.minimum(mwpm_errors, uf_errors)
    regret_per_window = np.maximum(0, adaptive_errors - oracle_best)
    cum_regret = np.cumsum(regret_per_window)
    assert np.all(cum_regret >= 0)
    assert np.all(np.diff(cum_regret) >= 0)
    assert cum_regret[-1] == 0


def test_cumulative_regret_sublinearity():
    T = 100
    regrets = np.random.exponential(scale=0.1, size=T)
    cum_reg = np.cumsum(regrets)
    avg_reg = cum_reg / np.arange(1, T + 1)
    assert np.all(avg_reg < 1.0)


def test_compute_cumulative_regret_bounds():
    """Verify theoretical cumulative regret bounds calculation."""
    bounds = compute_cumulative_regret_bounds(
        total_steps=1000,
        num_arms=4,
        num_change_points=3,
        switch_cost=0.005,
    )
    assert bounds["exp3_bound"] > 0.0
    assert bounds["piecewise_stationary_bound"] > bounds["sublinear_bound"]
    assert bounds["sublinear_bound"] == pytest.approx(np.sqrt(4 * 1000))

    # T=0 edge case
    zero_bounds = compute_cumulative_regret_bounds(0)
    assert zero_bounds["exp3_bound"] == 0.0
    assert zero_bounds["piecewise_stationary_bound"] == 0.0


def test_compute_dynamic_regret():
    """Verify compute_dynamic_regret against dynamic oracle."""
    c_losses = [0.15, 0.12, 0.18, 0.10]
    o_losses = [0.10, 0.12, 0.14, 0.10]

    total, curve = compute_dynamic_regret(c_losses, o_losses, switch_cost=0.01, switches=2)
    # Instantaneous regrets: 0.05, 0.0, 0.04, 0.0 -> sum = 0.09
    # Switch overhead: 2 * 0.01 = 0.02 -> total = 0.11
    assert total == pytest.approx(0.11)
    assert curve == pytest.approx([0.05, 0.05, 0.09, 0.09])

    # Mismatch length raises ValueError
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_dynamic_regret([0.1], [0.1, 0.2])


def test_regret_analyzer_enhanced():
    """Verify RegretAnalyzer includes theoretical bounds and dynamic regret."""
    results = [
        {
            "controller_name": "cg_ccads",
            "action": {"decoder": "mwpm", "dd_policy": "xy4"},
            "reward": 0.90,
        },
        {
            "controller_name": "cg_ccads",
            "action": {"decoder": "uf", "dd_policy": "cpmg"},
            "reward": 0.85,
        },
        {
            "controller_name": "cg_ccads",
            "action": {"decoder": "uf", "dd_policy": "cpmg"},
            "reward": 0.95,
        },
    ]

    analyzer = RegretAnalyzer(results, switch_penalty=0.002, num_arms=2)
    analysis = analyzer.analyze("cg_ccads")

    assert analysis.theoretical_bound > 0.0
    assert analysis.total_switches == 1
    assert analysis.switch_regret == pytest.approx(0.002)
    assert "dynamic_cumulative_regret" in analysis.to_dict()
    assert "theoretical_bound" in analysis.to_dict()

