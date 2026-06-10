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

    @property
    def has_budget(self) -> bool:
        return self.switches_in_current_epoch < self.max_switches_per_epoch


class MultiTimescaleEstimator:
    """
    Hierarchical multi-timescale noise estimator (TASK-028, Point 14).
    Fast EWMA for immediate alerts, CUSUM for change-point confirmation.
    """

    def __init__(
        self,
        alpha_fast: float = 0.35,
        alpha_slow: float = 0.05,
        cusum_threshold: float = 4.0,
        cusum_drift: float = 0.5,
    ) -> None:
        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow
        self.cusum_threshold = cusum_threshold
        self.cusum_drift = cusum_drift

        self.fast_rate: Optional[float] = None
        self.slow_rate: Optional[float] = None
        self.cusum_pos: float = 0.0
        self.cusum_neg: float = 0.0
        self.variance_estimate: float = 1e-4
        self.drift_detected: bool = False

    def update(self, defect_rate: float) -> tuple[float, bool]:
        """Update filters with new observed defect rate and return (D_t, is_change_point)."""
        if self.fast_rate is None:
            self.fast_rate = defect_rate
            self.slow_rate = defect_rate
            return 0.0, False

        # Update EWMA
        self.fast_rate = (1.0 - self.alpha_fast) * self.fast_rate + self.alpha_fast * defect_rate
        self.slow_rate = (1.0 - self.alpha_slow) * self.slow_rate + self.alpha_slow * defect_rate

        diff = self.fast_rate - self.slow_rate
        std = np.sqrt(max(self.variance_estimate, 1e-6))
        z_score = diff / std

        # CUSUM statistic
        self.cusum_pos = max(0.0, self.cusum_pos + z_score - self.cusum_drift)
        self.cusum_neg = max(0.0, self.cusum_neg - z_score - self.cusum_drift)
        drift_mag = max(self.cusum_pos, self.cusum_neg)

        is_change_point = drift_mag > self.cusum_threshold
        self.drift_detected = is_change_point

        # Online variance tracking
        self.variance_estimate = 0.95 * self.variance_estimate + 0.05 * (defect_rate - self.slow_rate) ** 2

        return float(drift_mag), is_change_point

    def reset_cusum(self) -> None:
        """Reset CUSUM accumulators post-adaptation."""
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.drift_detected = False


class CGCCADSController(BaseController):
    """
    Confidence-Gated Cost-Constrained Adaptive Decoder Selection (CG-CCADS).
    """

    def __init__(
        self,
        weights: Optional[CostWeights] = None,
        tax: Optional[AdaptationTax] = None,
        budget: Optional[AdaptationBudget] = None,
        confidence_level: float = 0.95,
        dead_zone_epsilon: float = 0.005,
        disagreement_threshold: float = 0.008,
        probation_blocks: int = 3,
        hysteresis_up: float = 0.08,
        hysteresis_down: float = 0.02,
        baseline_decoder: DecoderChoice = DecoderChoice.MWPM,
    ) -> None:
        super().__init__(weights=weights)
        self.tax = tax or AdaptationTax()
        self.budget = budget or AdaptationBudget()
        self.confidence_level = confidence_level
        self.z_score = 1.96 if confidence_level >= 0.95 else 1.645
        self.dead_zone_epsilon = dead_zone_epsilon
        self.disagreement_threshold = disagreement_threshold
        self.probation_blocks = probation_blocks
        self.hysteresis_up = hysteresis_up
        self.hysteresis_down = hysteresis_down
        self.baseline_decoder = baseline_decoder

        # Internal state
        self.estimator = MultiTimescaleEstimator()
        self.active_decoder: DecoderChoice = baseline_decoder
        self.candidate_decoder: DecoderChoice = DecoderChoice.UNION_FIND
        self.decision_state: ControllerDecision = ControllerDecision.STAY
        self.probation_counter: int = 0
        self.probation_performance_history: list[float] = []

        # Observation tracking
        self._current_state_vector: Optional[ControllerStateVector] = None
        self._last_defect_rates: collections.deque = collections.deque(maxlen=50)
        self._recent_surrogate_rewards: collections.deque = collections.deque(maxlen=20)
        self.total_switches: int = 0
        self.total_stays: int = 0
        self.total_rollbacks: int = 0

    @property
    def name(self) -> str:
        return "cg_ccads"

    def observe(self, state: HardwareState) -> None:
        """Ingest non-oracle observable state and update multi-timescale filters."""
        self._current_state = state
        defect_rate = getattr(state, "defect_rate", 0.05)
        self._last_defect_rates.append(defect_rate)

        drift_mag, _ = self.estimator.update(defect_rate)

        # Estimate disagreement Ct from empirical defect and calibration features
        # When defect rate is high or asymmetric, candidate decoders disagree more
        disagreement = min(0.15, max(0.0, 0.25 * abs(defect_rate - 0.03) + 0.01 * drift_mag))

        # Estimate performance difference delta_hat (surrogate gain of candidate vs active)
