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


@pytest.fixture
def dephasing_dominated_state() -> HardwareState:
    """Hardware state where T2 << T1 (dephasing dominant)."""
    return HardwareState(
        error_rate=0.008,
        t1_us=300.0,
        t2_us=50.0,   # Much shorter than T1
        readout_error=0.015,
        gate_error_1q=0.0008,
        gate_error_2q=0.005,
    )


@pytest.fixture
def relaxation_dominated_state() -> HardwareState:
    """Hardware state where T1 << T2 (relaxation dominant)."""
    return HardwareState(
        error_rate=0.008,
        t1_us=50.0,    # Much shorter than T2
        t2_us=300.0,
        readout_error=0.015,
        gate_error_1q=0.0008,
        gate_error_2q=0.005,
    )


@pytest.fixture
