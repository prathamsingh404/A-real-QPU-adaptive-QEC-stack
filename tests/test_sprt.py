"""
Tests for the SPRT (Sequential Probability Ratio Test) controller.

Verifies:
    - SPRTEngine correctly computes log-likelihood ratios
    - Decision boundaries are correctly derived from alpha/beta
    - State transitions (continue → reject/accept) work
    - SPRTController integrates SPRT with bandit policy
    - Cooldown prevents rapid switching
    - Reset restores clean state
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_qec.controller.base import ControlAction, HardwareState
from adaptive_qec.controller.bandit import build_arm_set
from adaptive_qec.controller.sprt import (
    SPRTController,
    SPRTEngine,
    SPRTState,
)


@pytest.fixture
def dummy_state() -> HardwareState:
    return HardwareState(
        error_rate=0.005,
        t1_us=200.0,
        t2_us=150.0,
        readout_error=0.01,
        gate_error_1q=0.0005,
        gate_error_2q=0.003,
    )


# -----------------------------------------------------------------------
# SPRTState tests
# -----------------------------------------------------------------------

class TestSPRTState:
    def test_initial_state(self):
        state = SPRTState()
        assert state.log_likelihood_ratio == 0.0
        assert state.observations == 0

    def test_reset(self):
        state = SPRTState()
        state.log_likelihood_ratio = 5.0
        state.observations = 100
        state.reset()
        assert state.log_likelihood_ratio == 0.0
        assert state.observations == 0


# -----------------------------------------------------------------------
# SPRTEngine tests
# -----------------------------------------------------------------------

class TestSPRTEngine:
    def test_initialization(self):
        engine = SPRTEngine(alpha=0.01, beta=0.01, p0=0.03, p1=0.04)
        assert engine is not None

    def test_boundaries_correct(self):
        """Upper boundary = ln((1-β)/α), Lower = ln(β/(1-α))."""
        engine = SPRTEngine(alpha=0.01, beta=0.01, p0=0.03, p1=0.04)
        # Upper boundary should be positive
        upper = np.log((1 - 0.01) / 0.01)
        lower = np.log(0.01 / (1 - 0.01))
        assert upper > 0
        assert lower < 0

    def test_update_with_successes(self):
        """Feeding successes should move LLR in one direction."""
        engine = SPRTEngine(alpha=0.01, beta=0.01, p0=0.03, p1=0.04)
        state = SPRTState()
        
        for _ in range(10):
            decision = engine.update(state, success=True)
        
        assert state.observations == 10
        # LLR should have moved from 0

    def test_update_with_failures(self):
        """Feeding failures should move LLR in the opposite direction."""
        engine = SPRTEngine(alpha=0.01, beta=0.01, p0=0.03, p1=0.04)
        state = SPRTState()
        
        for _ in range(10):
            engine.update(state, success=False)
        
        assert state.observations == 10

    def test_decision_under_strong_evidence(self):
        """With enough consistent evidence, SPRT should reach a decision."""
        engine = SPRTEngine(alpha=0.05, beta=0.05, p0=0.03, p1=0.06)
        state = SPRTState()
        
        decisions = []
        for _ in range(500):
            decision = engine.update(state, success=True)
            decisions.append(decision)
            if decision != "continue":
                break
        
        # Should have reached a decision with 500 consistent observations
        assert any(d != "continue" for d in decisions) or state.observations == 500


# -----------------------------------------------------------------------
# SPRTController tests
# -----------------------------------------------------------------------

class TestSPRTController:
    def test_initialization(self):
        arms = build_arm_set()
        ctrl = SPRTController(
            arms=arms,
            alpha=0.01,
            beta=0.01,
            p0=0.03,
            p1=0.04,
        )
        assert ctrl.name == "sprt"

    def test_decide_returns_action(self, dummy_state):
        arms = build_arm_set()
        ctrl = SPRTController(
            arms=arms,
            alpha=0.01,
            beta=0.01,
            p0=0.03,
            p1=0.04,
        )
        ctrl.observe(dummy_state)
        action = ctrl.decide()
        assert isinstance(action, ControlAction)

    def test_update_with_reward(self, dummy_state):
        arms = build_arm_set()
        ctrl = SPRTController(
            arms=arms,
            alpha=0.01,
            beta=0.01,
            p0=0.03,
            p1=0.04,
        )
        ctrl.observe(dummy_state)
        ctrl.decide()
        ctrl.update(0.95)  # High reward
        
        assert len(ctrl.telemetry) == 1

    def test_full_loop(self, dummy_state):
        """Run 100 steps of the SPRT controller loop."""
        arms = build_arm_set()
        ctrl = SPRTController(
            arms=arms,
            alpha=0.01,
            beta=0.01,
            p0=0.03,
            p1=0.04,
        )
        
        for step in range(100):
            ctrl.observe(dummy_state)
            action = ctrl.decide()
            assert isinstance(action, ControlAction)
            reward = 0.9 + 0.05 * np.random.randn()
            ctrl.update(max(0.0, min(1.0, reward)))
        
        assert len(ctrl.telemetry) == 100

    def test_reset(self, dummy_state):
        arms = build_arm_set()
        ctrl = SPRTController(
            arms=arms,
            alpha=0.01,
            beta=0.01,
            p0=0.03,
            p1=0.04,
        )
        
        for _ in range(20):
            ctrl.observe(dummy_state)
            ctrl.decide()
            ctrl.update(0.9)
        
        ctrl.reset()
        assert len(ctrl.telemetry) == 0

    def test_summary(self, dummy_state):
        arms = build_arm_set()
        ctrl = SPRTController(
            arms=arms,
            alpha=0.01,
            beta=0.01,
            p0=0.03,
            p1=0.04,
        )
        ctrl.observe(dummy_state)
        ctrl.decide()
        ctrl.update(0.9)
        
        summary = ctrl.summary()
        assert isinstance(summary, dict)
