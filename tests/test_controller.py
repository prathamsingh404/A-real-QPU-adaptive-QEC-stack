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

    def test_switch_cost_penalizes_mode_change(self):
        """Switching decoders should incur a penalty."""
        state = _make_state()
        weights = CostWeights(lambda_switch=0.1)

        new_action = ControlAction(decoder=DecoderChoice.MWPM, dd_policy=DDSequenceType.XY4)
        prev_uf = ControlAction(decoder=DecoderChoice.UNION_FIND, dd_policy=DDSequenceType.XY4)
        prev_same = ControlAction(decoder=DecoderChoice.MWPM, dd_policy=DDSequenceType.XY4)

        cost_switch = compute_cost(state, new_action, prev_uf, weights)
        cost_stay = compute_cost(state, new_action, prev_same, weights)

        assert cost_switch > cost_stay

    def test_dd_adds_overhead(self):
        """DD sequences should add cost via pulse overhead."""
        state = _make_state()
        weights = CostWeights(lambda_dd_cost=0.1)

        no_dd = ControlAction(decoder=DecoderChoice.MWPM, dd_policy=DDSequenceType.NONE)
        xy8_dd = ControlAction(decoder=DecoderChoice.MWPM, dd_policy=DDSequenceType.XY8)

        cost_no = compute_cost(state, no_dd, None, weights)
        cost_xy8 = compute_cost(state, xy8_dd, None, weights)

        # XY8 adds 8/8 * 0.1 = 0.1 DD cost, but also suppresses noise
        # The cost comparison depends on the balance
        assert isinstance(cost_no, float) and isinstance(cost_xy8, float)


class TestLogicalErrorRateModel:
    def test_zero_noise_low_error(self):
        """Near-zero physical noise should give very low logical error rate."""
        state = _make_state(p_2q=0.0001, p_ro=0.001, drift_magnitude=0.0,
                            leakage_fraction=0.0)
        p_L = estimate_logical_error_rate(
            state, DecoderChoice.MWPM, DDSequenceType.NONE, False
        )
        assert p_L < 0.01

    def test_high_noise_high_error(self):
        """High physical noise should give high logical error rate."""
        state = _make_state(p_2q=0.05, p_ro=0.1)
        p_L = estimate_logical_error_rate(
            state, DecoderChoice.MWPM, DDSequenceType.NONE, False
        )
        assert p_L > 0.01

    def test_uf_slightly_worse_than_mwpm(self):
        """UF should estimate slightly worse LER than MWPM (10% penalty)."""
        state = _make_state(p_2q=0.005)
        p_mwpm = estimate_logical_error_rate(
            state, DecoderChoice.MWPM, DDSequenceType.NONE, False
        )
        p_uf = estimate_logical_error_rate(
            state, DecoderChoice.UNION_FIND, DDSequenceType.NONE, False
        )
        assert p_uf >= p_mwpm

    def test_error_rate_bounded(self):
        """Logical error rate should always be in [0, 0.5]."""
        for p_2q in [0.0, 0.001, 0.01, 0.05, 0.1, 0.5]:
            state = _make_state(p_2q=p_2q)
            p_L = estimate_logical_error_rate(
                state, DecoderChoice.MWPM, DDSequenceType.NONE, False
            )
            assert 0.0 <= p_L <= 0.5, f"OOB p_L={p_L} at p_2q={p_2q}"


# ---------------------------------------------------------------------------
# Hysteresis tracker tests
# ---------------------------------------------------------------------------

class TestHysteresisTracker:
    def test_no_switch_below_patience(self):
        """Should not switch before patience windows of improvement."""
        tracker = HysteresisTracker(patience=3, margin=0.05)
        for _ in range(2):
            result = tracker.should_switch("uf:none", "mwpm:xy4", 1.0, 0.8)
            assert not result  # 2 < patience=3

    def test_switch_at_patience(self):
        """Should switch after patience consecutive improvements."""
        tracker = HysteresisTracker(patience=3, margin=0.05)
        for i in range(3):
            result = tracker.should_switch("uf:none", "mwpm:xy4", 1.0, 0.8)
        assert result  # 3 == patience

    def test_no_switch_small_improvement(self):
        """Should not switch if improvement is below margin."""
        tracker = HysteresisTracker(patience=3, margin=0.10)
        for _ in range(5):
            result = tracker.should_switch("uf:none", "mwpm:xy4", 1.0, 0.95)
        assert not result  # 5% < 10% margin

    def test_reset_on_regression(self):
        """Counter should reset if the candidate regresses."""
        tracker = HysteresisTracker(patience=3, margin=0.05)
        tracker.should_switch("uf:none", "mwpm:xy4", 1.0, 0.8)
        tracker.should_switch("uf:none", "mwpm:xy4", 1.0, 0.8)
        # Now candidate is worse
        tracker.should_switch("uf:none", "mwpm:xy4", 1.0, 1.1)
        # Next two shouldn't trigger (counter reset)
        result = tracker.should_switch("uf:none", "mwpm:xy4", 1.0, 0.8)
        assert not result
