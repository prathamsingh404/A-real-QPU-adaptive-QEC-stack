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
