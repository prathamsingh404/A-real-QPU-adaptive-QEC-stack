"""
Tests for the bandit controller family (Exp3, Exp3.P, DA-SE).

Verifies:
    - Correct initialization and arm count
    - Probability distribution sums to 1
    - Reward updates shift probability mass correctly
    - Exp3 importance-weighted updates are numerically stable
    - DA-SE arm elimination and reactivation under drift
    - Factory function produces correct controller types
    - Convergence to best arm under stationary rewards
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_qec.controller.base import ControlAction, HardwareState
from adaptive_qec.controller.bandit import (
    BanditArm,
    DASEController,
    Exp3Controller,
    Exp3PController,
    build_arm_set,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def arms() -> list[BanditArm]:
    """Standard 6-arm set."""
    return build_arm_set()


@pytest.fixture
def dummy_state() -> HardwareState:
    """Minimal hardware state for testing."""
    return HardwareState(
        error_rate=0.005,
        t1_us=200.0,
        t2_us=150.0,
        readout_error=0.01,
        gate_error_1q=0.0005,
        gate_error_2q=0.003,
    )


# -----------------------------------------------------------------------
# BanditArm tests
# -----------------------------------------------------------------------

class TestBanditArm:
    def test_arm_label(self):
        arm = BanditArm(decoder="mwpm", dd_sequence="xy4", schedule="balanced")
        assert "mwpm" in arm.label
        assert "xy4" in arm.label

    def test_build_arm_set_default(self):
        arms = build_arm_set()
        assert len(arms) == 6
        decoders = {a.decoder for a in arms}
        assert "mwpm" in decoders
        assert "union_find" in decoders

    def test_arm_uniqueness(self):
        arms = build_arm_set()
        labels = [a.label for a in arms]
        assert len(labels) == len(set(labels))


# -----------------------------------------------------------------------
# Exp3Controller tests
# -----------------------------------------------------------------------

class TestExp3Controller:
    def test_initialization(self, arms):
        ctrl = Exp3Controller(arms=arms, gamma=0.1)
        assert ctrl.name == "exp3"

    def test_decide_returns_valid_action(self, arms, dummy_state):
        ctrl = Exp3Controller(arms=arms, gamma=0.1)
        ctrl.observe(dummy_state)
        action = ctrl.decide()
        assert isinstance(action, ControlAction)

    def test_probability_distribution_sums_to_one(self, arms):
        ctrl = Exp3Controller(arms=arms, gamma=0.1)
        probs = ctrl._compute_probs()
        assert abs(probs.sum() - 1.0) < 1e-10
        assert all(p >= 0 for p in probs)

    def test_update_shifts_weights(self, arms, dummy_state):
        ctrl = Exp3Controller(arms=arms, gamma=0.1)
        ctrl.observe(dummy_state)
        action = ctrl.decide()

        probs_before = ctrl._compute_probs().copy()
        ctrl.update(1.0)  # High reward for chosen arm
        probs_after = ctrl._compute_probs()

        # Distribution should have shifted
        assert not np.allclose(probs_before, probs_after)

    def test_numerical_stability_under_extreme_rewards(self, arms, dummy_state):
        ctrl = Exp3Controller(arms=arms, gamma=0.01)
        for _ in range(500):
            ctrl.observe(dummy_state)
            ctrl.decide()
            ctrl.update(1.0)
        
        probs = ctrl._compute_probs()
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))
        assert abs(probs.sum() - 1.0) < 1e-8

    def test_reset_restores_uniform(self, arms, dummy_state):
        ctrl = Exp3Controller(arms=arms, gamma=0.1)
        for _ in range(20):
            ctrl.observe(dummy_state)
            ctrl.decide()
            ctrl.update(0.5)
        ctrl.reset()
        probs = ctrl._compute_probs()
        expected = 1.0 / len(arms)
        assert all(abs(p - expected) < 0.01 for p in probs)


# -----------------------------------------------------------------------
# Exp3PController tests
# -----------------------------------------------------------------------
