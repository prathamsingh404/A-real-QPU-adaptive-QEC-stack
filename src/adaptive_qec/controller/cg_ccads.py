"""
Confidence-Gated Cost-Constrained Adaptive Decoder Selection (CG-CCADS).

Core novel controller architecture for adaptive quantum error correction
(TASK-026 through TASK-048, Submission Gate B).

Mathematical Architecture:
--------------------------
1. Controller State Representation (TASK-027):
       x_t = [p_hat_X, p_hat_Z, p_hat_2q, r_hat_det, D_t, C_t, U_t, G_t]
   where:
       p_hat_X, p_hat_Z: Estimated X and Z stabilizer defect rates
       p_hat_2q: Inferred or calibrated two-qubit gate error rate
       r_hat_det: Total detector firing density
       D_t: Multi-timescale drift magnitude (EWMA + CUSUM change-point score)
       C_t: Pairwise candidate decoder disagreement rate
       U_t: Standard error / uncertainty of performance difference
       G_t: Net estimated performance gain delta_hat

2. Multi-Timescale Estimator (TASK-028, TASK-047):
       Short-term: EWMA detector rate filter (fast anomaly alerts)
       Medium-term: CUSUM / SPRT change-point detector for drift significance
       Long-term: Sliding-window covariance and stationary reference tracking

3. Three-Gate Decision Pipeline (TASK-046):
       Gate 1 (Drift Evidence): If D_t <= T_drift, noise is stationary -> STAY (Cost = 0).
       Gate 2 (Disagreement): If C_t <= delta_disagree, candidate decoders agree on >99.5%
              of corrections -> STAY (Adapting produces negligible observable change).
       Gate 3 (Cost-Constrained LCB): Evaluate lower confidence bound of gain:
              LCB(delta) = delta_hat - z_{1-alpha} * sigma_delta
              If LCB(delta) > tau_adapt + epsilon_deadzone:
                  SWITCH to candidate under probationary period of K blocks.
              Else:
                  STAY.

4. Adaptation Tax Model (TASK-041):
       tau_adapt = C_switch + C_latency + C_compute + C_physical

5. Safety Safeguards:
       - Explicit STAY (zero cost, deliberate abstention) (TASK-030)
       - Dead zone (|delta_hat| < epsilon -> STAY) (TASK-038)
       - Dual-threshold hysteresis (T_up > T_down) (TASK-035)
       - Probationary window (K blocks) with automatic ROLLBACK (TASK-036, TASK-037)
       - Hard AdaptationBudget constraint per epoch (TASK-040)
       - Deployable surrogate objective without unobservable true LER (TASK-031, TASK-032)
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from adaptive_qec.controller.base import (
    BaseController,
    ControlAction,
    HardwareState,
    InterventionType,
)
from adaptive_qec.controller.controller import CostWeights, DecoderChoice
from adaptive_qec.mitigation.dynamical_decoupling import DDSequenceType

logger = logging.getLogger(__name__)


class ControllerDecision(str, Enum):
    """First-class CG-CCADS decision outcome."""
    STAY = "stay"          # Deliberate abstention, 0 cost
    SWITCH = "switch"      # Commit to candidate decoder under LCB constraint
    PROBATION = "probation"# Trial period for candidate decoder
    ROLLBACK = "rollback"  # Automatic retreat to baseline upon degradation


@dataclass
class ControllerStateVector:
    """Mathematical state representation x_t (TASK-027, Point 13)."""
    p_hat_x: float = 0.0
    p_hat_z: float = 0.0
    p_hat_2q: float = 0.0
    r_hat_det: float = 0.0
    drift_magnitude: float = 0.0       # D_t
    decoder_disagreement: float = 0.0  # C_t
    uncertainty: float = 0.0           # U_t
    estimated_gain: float = 0.0        # G_t

    def to_array(self) -> np.ndarray:
        return np.array([
            self.p_hat_x,
            self.p_hat_z,
            self.p_hat_2q,
            self.r_hat_det,
            self.drift_magnitude,
            self.decoder_disagreement,
            self.uncertainty,
            self.estimated_gain,
        ], dtype=np.float64)


@dataclass
class AdaptationTax:
    """Physical and computational cost model tau_adapt (TASK-041, Point 59)."""
    c_switch: float = 0.015    # Penalty for switching decoder modes / invalidating cache
    c_latency: float = 0.010   # Execution latency tax (e.g. UF vs MWPM graph build)
    c_compute: float = 0.005   # Re-weighting or graph solve overhead
    c_physical: float = 0.0    # Circuit pulse overhead (0 for classical-only actions)

    @property
    def total(self) -> float:
        return self.c_switch + self.c_latency + self.c_compute + self.c_physical


@dataclass
class AdaptationBudget:
    """Explicit switching constraint to prevent thrashing (TASK-040, Point 58)."""
    max_switches_per_epoch: int = 4
    epoch_length: int = 100
    switches_in_current_epoch: int = 0
    epoch_step: int = 0

    def record_step(self, switched: bool) -> bool:
        """Update step and check if adaptation budget allows a switch."""
        self.epoch_step += 1
        if self.epoch_step >= self.epoch_length:
            self.epoch_step = 0
            self.switches_in_current_epoch = 0

        if switched:
            if self.switches_in_current_epoch >= self.max_switches_per_epoch:
                return False  # Budget exhausted
            self.switches_in_current_epoch += 1
        return True
