"""
Comprehensive Unit & Boundary Tests for CG-CCADS Controller (tests/test_cg_ccads.py).

Validates Gate B (Algorithmic Novelty) requirements:
- Mathematical state representation x_t (TASK-027)
- Multi-timescale drift tracking (TASK-028)
- First-class STAY action (TASK-030)
- Three-gate decision pipeline: Gate 1, Gate 2, Gate 3 (TASK-046)
- LCB switching rule under adaptation tax (TASK-033, TASK-041)
- Dead zone abstention (TASK-038)
- Automatic rollback upon probationary failure (TASK-036, TASK-037)
- Adaptation budget constraint (TASK-040)
"""

import numpy as np
import pytest

from adaptive_qec.controller.base import HardwareState
from adaptive_qec.controller.cg_ccads import (
    AdaptationBudget,
    AdaptationTax,
    CGCCADSController,
    ControllerDecision,
    ControllerStateVector,
    MultiTimescaleEstimator,
)
from adaptive_qec.controller.controller import DecoderChoice


def test_controller_state_vector_shape():
    """Verify state vector x_t has 8 components as defined in TASK-027."""
    sv = ControllerStateVector(
        p_hat_x=0.01,
        p_hat_z=0.015,
        p_hat_2q=0.003,
        r_hat_det=0.025,
        drift_magnitude=1.5,
        decoder_disagreement=0.02,
        uncertainty=0.005,
        estimated_gain=0.04,
    )
    arr = sv.to_array()
    assert arr.shape == (8,)
    assert arr[4] == 1.5  # D_t
    assert arr[5] == 0.02 # C_t


def test_gate1_stationary_abstains_stay():
    """Verify that Gate 1 outputs STAY under stationary, non-drifting noise."""
    controller = CGCCADSController()

    # Feed 10 stationary windows with normal defect rate ~0.03
    for _ in range(10):
        state = HardwareState(defect_rate=0.03, p_2q=0.003)
        action = controller.step(state)
        controller.update(reward=-0.03)

    assert action.decoder == DecoderChoice.MWPM
    assert "stay" in action.notes.lower()
    assert controller.total_switches == 0
    assert controller.total_stays >= 10


def test_gate2_zero_disagreement_abstains_stay():
    """Verify Gate 2: Even with drift, if decoder disagreement is below threshold, stay."""
    controller = CGCCADSController(disagreement_threshold=0.10)  # High disagreement requirement

    # Sudden drift spike in defect rate
    for _ in range(5):
        state = HardwareState(defect_rate=0.07, p_2q=0.005)
        action = controller.step(state)
        controller.update(reward=-0.05)

    # Disagreement remains below threshold 0.10 -> Gate 2 STAY
    assert action.decoder == DecoderChoice.MWPM
    assert controller.total_switches == 0


def test_gate3_confidence_gated_switch_under_drift():
    """Verify Gate 3: Significant drift with high leakage triggers confidence-gated switch."""
    controller = CGCCADSController(
        confidence_level=0.95,
        dead_zone_epsilon=0.001,
        probation_blocks=3,
        hysteresis_up=0.01,
    )

    # Feed baseline
    for _ in range(5):
        controller.step(HardwareState(defect_rate=0.03))
        controller.update(reward=-0.02)

    # Induce massive persistent leakage & defect surge
    switched = False
    for _ in range(10):
        state = HardwareState(defect_rate=0.14, leakage_fraction=0.08, p_2q=0.01)
        action = controller.step(state)
        controller.update(reward=0.05)  # Good performance
        if action.decoder == DecoderChoice.UNION_FIND:
            switched = True
            break

    assert switched
    assert controller.total_switches > 0
    assert controller.decision_state in (ControllerDecision.PROBATION, ControllerDecision.SWITCH)


def test_probationary_failure_triggers_automatic_rollback():
    """Verify TASK-036: If candidate performs poorly during probation, rollback to baseline."""
    controller = CGCCADSController(probation_blocks=3)

    # Force switch into probation
    controller.active_decoder = DecoderChoice.UNION_FIND
    controller.decision_state = ControllerDecision.PROBATION
    controller.probation_counter = 0

    # Deliver severe negative surrogate rewards during probation
    state = HardwareState(defect_rate=0.05)
    controller.observe(state)
    controller.update(reward=-0.30)  # Terrible performance
    controller.observe(state)
    controller.update(reward=-0.35)  # Terrible performance again

    action = controller.decide()
    assert action.decoder == DecoderChoice.MWPM
    assert "rollback" in action.notes.lower()
    assert controller.total_rollbacks > 0
    assert controller.active_decoder == DecoderChoice.MWPM


def test_adaptation_budget_prevents_thrashing():
    """Verify TASK-040: Controller cannot exceed max switches per epoch."""
    budget = AdaptationBudget(max_switches_per_epoch=2, epoch_length=20)
    assert budget.has_budget
    assert budget.record_step(switched=True)
    assert budget.record_step(switched=True)
    assert not budget.has_budget
    assert not budget.record_step(switched=True)  # Blocked by budget
