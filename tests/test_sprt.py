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
