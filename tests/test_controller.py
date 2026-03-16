"""Tests for the adaptive controller.

Validates the core s_t -> a_t -> cost loop, hysteresis behavior,
and the interpretability of the cost function.
"""

import numpy as np
import pytest

from adaptive_qec.controller.controller import (
    AdaptiveController,
    ControlAction,
    CostWeights,
    DecoderChoice,
    HardwareState,
    HysteresisTracker,
    compute_cost,
    estimate_logical_error_rate,
)
from adaptive_qec.mitigation.dynamical_decoupling import DDSequenceType
from adaptive_qec.noise.drift import DriftStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> HardwareState:
    """Create a HardwareState with reasonable defaults."""
    defaults = dict(
        defect_rate=0.05,
        drift_magnitude=0.5,
        drift_status=DriftStatus.STABLE,
        burst_active=False,
        leakage_fraction=0.002,
        t1_mean_us=150.0,
        t2_mean_us=120.0,
        p_1q=0.0003,
        p_2q=0.005,
        p_ro=0.01,
        code_distance=3,
        num_data_qubits=9,
        num_detectors=8,
    )
    defaults.update(overrides)
    return HardwareState(**defaults)


# ---------------------------------------------------------------------------
# HardwareState tests
# ---------------------------------------------------------------------------

class TestHardwareState:
    def test_to_vector_shape(self):
        state = _make_state()
        vec = state.to_vector()
        assert vec.shape == (10,)
        assert vec.dtype == np.float64

    def test_to_vector_values_normalized(self):
        """Feature vector should have values roughly in [0, 2] range."""
        state = _make_state()
        vec = state.to_vector()
        assert np.all(np.abs(vec) < 10.0), f"Unnormalized features: {vec}"


# ---------------------------------------------------------------------------
# Cost function tests
# ---------------------------------------------------------------------------

class TestCostFunction:
    def test_lower_noise_lower_cost(self):
        """Lower physical noise should produce lower cost."""
        low_noise = _make_state(p_2q=0.001)
        high_noise = _make_state(p_2q=0.02)

        action = ControlAction(decoder=DecoderChoice.MWPM, dd_policy=DDSequenceType.NONE)
        weights = CostWeights()

        cost_low = compute_cost(low_noise, action, None, weights)
        cost_high = compute_cost(high_noise, action, None, weights)

        assert cost_low < cost_high

    def test_burst_mitigation_helps_during_burst(self):
        """Burst mitigation should reduce cost when a burst is active."""
        state = _make_state(burst_active=True, p_2q=0.005)
        weights = CostWeights()

        no_mit = ControlAction(
            decoder=DecoderChoice.MWPM,
            dd_policy=DDSequenceType.NONE,
            burst_mitigation=False,
        )
        mit = ControlAction(
            decoder=DecoderChoice.MWPM,
            dd_policy=DDSequenceType.NONE,
            burst_mitigation=True,
        )

        cost_no = compute_cost(state, no_mit, None, weights)
        cost_mit = compute_cost(state, mit, None, weights)

        assert cost_mit < cost_no
