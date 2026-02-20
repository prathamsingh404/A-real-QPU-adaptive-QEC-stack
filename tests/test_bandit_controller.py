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
