"""
Shared pytest fixtures and configuration for AdaptiveQEC test suite.
"""

from __future__ import annotations

import numpy as np
import pytest
import stim

from adaptive_qec.controller.base import ControlAction, HardwareState
from adaptive_qec.controller.bandit import BanditArm, build_arm_set


# -----------------------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def dummy_hardware_state() -> HardwareState:
    """Typical IBM Heron hardware state."""
    return HardwareState(
        error_rate=0.005,
        t1_us=200.0,
        t2_us=150.0,
        readout_error=0.01,
        gate_error_1q=0.0005,
        gate_error_2q=0.003,
    )
