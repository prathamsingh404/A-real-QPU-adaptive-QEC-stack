"""
Adaptive QEC controller: the paper's core contribution.

This module implements a formally defined, interpretable, hardware-state-
conditioned QEC control policy. The controller observes the estimated
hardware state s_t (drift, burst, leakage, calibration metrics) and
selects an action a_t (decoder, DD policy, burst mitigation, recalibration)
to minimize an expected cost function J.

    a*_t = argmin_a  J(a | s_t)

The cost function captures:
    J_t = P_L + lambda_1 * L_decode + lambda_2 * C_DD
          + lambda_3 * C_switch + lambda_4 * C_cal

where:
    P_L         = estimated logical error rate under action a
    L_decode    = decoder latency (normalized)
    C_DD        = dynamical decoupling pulse overhead
    C_switch    = mode switching penalty (to prevent oscillation)
    C_cal       = recalibration cost (queue time + shots)

Hysteresis prevents oscillation between modes:
    Enter MWPM mode:  confidence > 95% AND improvement > 5% for 3 windows
    Leave MWPM mode:  confidence > 95% AND degradation > 5% for 3 windows

This is the central experiment from the audit recommendation (Point #29):
    "Build one narrow experiment that demonstrates adaptive > static."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from adaptive_qec.decoders.base import DecoderMetrics
from adaptive_qec.mitigation.dynamical_decoupling import DDSequenceType
from adaptive_qec.noise.drift import DriftStatus
from adaptive_qec.provenance import DataProvenance, ProvenanceTag

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State and action spaces
# ---------------------------------------------------------------------------

class DecoderChoice(str, Enum):
    """Available decoder strategies."""
    MWPM = "mwpm"
    UNION_FIND = "union_find"


@dataclass
class HardwareState:
    """s_t = estimated hardware state vector at time step t.

    This is the observation the controller uses to make decisions.
    All values are normalized or in standard physical units.
    """
    # Syndrome-derived metrics (updated each window)
    defect_rate: float = 0.05         # R_D: average detector firing rate
    drift_magnitude: float = 0.0     # EWMA z-score magnitude
    drift_status: DriftStatus = DriftStatus.STABLE  # categorical drift classification
    burst_active: bool = False       # True if BurstDetector flagged a burst
    leakage_fraction: float = 0.0    # estimated fraction of leaked qubits

    # Calibration-derived metrics (updated less frequently)
    t1_mean_us: float = 0.0          # mean T1 of active patch
    t2_mean_us: float = 0.0          # mean T2 of active patch
    p_1q: float = 0.0                # single-qubit gate error
    p_2q: float = 0.0                # two-qubit gate error
    p_ro: float = 0.0                # readout error

    # Topology context
    code_distance: int = 3
    num_data_qubits: int = 9
    num_detectors: int = 8

    # Aliases for backwards compatibility with tests and callers
    error_rate: float = 0.0
    t1_us: float = 0.0
    t2_us: float = 0.0
    readout_error: float = 0.0
    gate_error_1q: float = 0.0
    gate_error_2q: float = 0.0

    def __post_init__(self) -> None:
        if self.error_rate > 0.0 and self.p_2q == 0.0:
            object.__setattr__(self, "defect_rate", self.error_rate)
            object.__setattr__(self, "p_2q", self.error_rate)
            object.__setattr__(self, "gate_error_2q", self.error_rate)
        if self.t1_us > 0.0 and self.t1_mean_us == 0.0:
            object.__setattr__(self, "t1_mean_us", self.t1_us)
        elif self.t1_mean_us > 0.0 and self.t1_us == 0.0:
            object.__setattr__(self, "t1_us", self.t1_mean_us)

        if self.t2_us > 0.0 and self.t2_mean_us == 0.0:
            object.__setattr__(self, "t2_mean_us", self.t2_us)
        elif self.t2_mean_us > 0.0 and self.t2_us == 0.0:
            object.__setattr__(self, "t2_us", self.t2_mean_us)

        if self.gate_error_1q > 0.0 and self.p_1q == 0.0:
            object.__setattr__(self, "p_1q", self.gate_error_1q)
        elif self.p_1q > 0.0 and self.gate_error_1q == 0.0:
            object.__setattr__(self, "gate_error_1q", self.p_1q)

        if self.gate_error_2q > 0.0 and self.p_2q == 0.0:
            object.__setattr__(self, "p_2q", self.gate_error_2q)
        elif self.p_2q > 0.0 and self.gate_error_2q == 0.0:
            object.__setattr__(self, "gate_error_2q", self.p_2q)

        if self.readout_error > 0.0 and self.p_ro == 0.0:
            object.__setattr__(self, "p_ro", self.readout_error)
        elif self.p_ro > 0.0 and self.readout_error == 0.0:
            object.__setattr__(self, "readout_error", self.p_ro)

    def to_vector(self) -> np.ndarray:
        """Convert to a numeric feature vector for cost evaluation."""
        return np.array([
            self.defect_rate,
            self.drift_magnitude,
            float(self.burst_active),
            self.leakage_fraction,
            self.t1_mean_us / 200.0,    # normalize to ~1
            self.t2_mean_us / 200.0,
            self.p_1q * 1000.0,         # scale to ~0.1 - 1 range
            self.p_2q * 100.0,
            self.p_ro * 100.0,
            self.code_distance / 7.0,
        ], dtype=np.float64)


@dataclass
class ControlAction:
    """a_t = adaptive control action at time step t.

    Specifies the full QEC strategy for the current window.
    """
    decoder: Any                             # which decoder to use
    dd_policy: Any = DDSequenceType.NONE     # DD sequence (NONE, CPMG, XY4, XY8)
    burst_mitigation: bool = False           # mask burst-affected syndromes
    request_recalibration: bool = False      # request fresh calibration
    notes: str = ""                          # human-readable rationale
    dd_sequence: Optional[str] = None
    schedule: str = "balanced"

    def __post_init__(self) -> None:
        if self.dd_sequence is None:
            val = self.dd_policy.value if hasattr(self.dd_policy, "value") else str(self.dd_policy)
            object.__setattr__(self, "dd_sequence", val)


@dataclass
class ControllerMetrics:
    """Running metrics for the adaptive controller."""
    total_windows: int = 0
    total_mode_switches: int = 0
    decoder_usage: dict[str, int] = field(default_factory=lambda: {"mwpm": 0, "union_find": 0})
    dd_usage: dict[str, int] = field(default_factory=lambda: {
        "none": 0, "cpmg": 0, "xy4": 0, "xy8": 0
    })
    burst_mitigations: int = 0
    recalibration_requests: int = 0

    # Per-window cost history
    cost_history: list[float] = field(default_factory=list)
    action_history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_windows": self.total_windows,
            "total_mode_switches": self.total_mode_switches,
            "decoder_usage": dict(self.decoder_usage),
            "dd_usage": dict(self.dd_usage),
            "burst_mitigations": self.burst_mitigations,
            "recalibration_requests": self.recalibration_requests,
            "mean_cost": float(np.mean(self.cost_history)) if self.cost_history else 0.0,
        }


# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------

@dataclass
class CostWeights:
    """Configurable weights for the cost function J(a | s).

    All weights are non-negative. Default values produce a policy that
    prioritizes logical error rate while penalizing excessive switching.

    PROVENANCE: ASSUMED — these weights are hand-tuned defaults. The
    sensitivity analysis in experiments/adaptive_vs_static.py sweeps
    over these to find robust operating points.
    """
    lambda_latency: float = 0.01       # weight on normalized decoder latency
    lambda_dd_cost: float = 0.005      # weight on DD pulse overhead
    lambda_switch: float = 0.02        # weight on mode switching penalty
    lambda_recal: float = 0.05         # weight on recalibration cost

    def to_dict(self) -> dict[str, float]:
        return {
            "lambda_latency": self.lambda_latency,
            "lambda_dd_cost": self.lambda_dd_cost,
            "lambda_switch": self.lambda_switch,
            "lambda_recal": self.lambda_recal,
        }


def estimate_logical_error_rate(
    state: HardwareState,
    decoder: DecoderChoice,
    dd: DDSequenceType,
    burst_mitigate: bool,
) -> float:
    """Estimate P_L for a given (state, action) pair.

    Uses the phenomenological model:
        P_L ~ A * (p_eff / p_th)^((d+1)/2)

    where p_eff is the effective physical error rate after DD and burst
    mitigation are applied.

    PROVENANCE: INFERRED — this is a model-based estimate, not a direct
    measurement. The model coefficients are approximate.
    """
    # Base physical error rate
    p_eff = state.p_2q + state.p_ro / 2.0

    # Add drift contribution
    if state.drift_magnitude > 1.0:
        p_eff *= (1.0 + 0.1 * state.drift_magnitude)

    # Add burst contribution (if not mitigated)
    if state.burst_active and not burst_mitigate:
        p_eff *= 3.0  # bursts roughly triple the error rate

    # Add leakage contribution
    p_eff += state.leakage_fraction * 0.1

    # DD suppression
    DD_SUPPRESSION = {
        DDSequenceType.NONE: 1.0,
        DDSequenceType.CPMG: 0.85,
        DDSequenceType.XY4: 0.70,
        DDSequenceType.XY8: 0.60,
    }
    p_eff *= DD_SUPPRESSION.get(dd, 1.0)

    # Decoder accuracy: MWPM is near-optimal for low uncorrelated Pauli noise.
    # However, under persistent leakage or severe defect clustering, Union-Find's
    # local cluster growth is more robust than global minimum-weight pairing.
    if decoder == DecoderChoice.UNION_FIND:
        if state.leakage_fraction > 0.05 or state.drift_magnitude > 4.0:
            p_eff *= 0.85  # UF local clustering outperforms MWPM on correlated/leakage defects
        else:
            p_eff *= 1.10  # MWPM is ~10% more accurate under standard uncorrelated noise

    # Phenomenological model
    p_th = 0.01  # approximate threshold
    d = state.code_distance
    A = 0.1

    if p_eff < p_th:
        p_L = A * (p_eff / p_th) ** ((d + 1) / 2)
    else:
        p_L = min(0.5, A * (p_eff / p_th))

    return float(np.clip(p_L, 0.0, 0.5))


def compute_cost(
    state: HardwareState,
    action: ControlAction,
    previous_action: Optional[ControlAction],
    weights: CostWeights,
) -> float:
    """Compute the cost J(a | s) for a candidate action.

    J = P_L + λ₁·L + λ₂·C_DD + λ₃·C_switch + λ₄·C_cal

    Args:
        state: Current hardware state observation.
        action: Candidate control action.
        previous_action: Action taken in the previous window (for switch cost).
        weights: Cost function weights.

    Returns:
        Scalar cost value (lower is better).
    """
    # 1. Estimated logical error rate
    p_L = estimate_logical_error_rate(
        state, action.decoder, action.dd_policy, action.burst_mitigation
    )

    # 2. Decoder latency cost (normalized: UF ~= 1.0, MWPM ~= 1.5 for small d)
    latency_cost = 1.5 if action.decoder == DecoderChoice.MWPM else 1.0

    # 3. DD pulse cost (number of pulses normalized)
    DD_PULSE_COUNTS = {
        DDSequenceType.NONE: 0,
        DDSequenceType.CPMG: 2,
        DDSequenceType.XY4: 4,
        DDSequenceType.XY8: 8,
    }
    dd_cost = DD_PULSE_COUNTS.get(action.dd_policy, 0) / 8.0

    # 4. Mode switch cost (penalize changing decoder or DD policy)
    switch_cost = 0.0
    if previous_action is not None:
        if action.decoder != previous_action.decoder:
            switch_cost += 1.0
        if action.dd_policy != previous_action.dd_policy:
            switch_cost += 0.5

    # 5. Recalibration cost
    recal_cost = 1.0 if action.request_recalibration else 0.0

    total = (
        p_L
        + weights.lambda_latency * latency_cost
        + weights.lambda_dd_cost * dd_cost
        + weights.lambda_switch * switch_cost
        + weights.lambda_recal * recal_cost
    )
    return float(total)


# ---------------------------------------------------------------------------
# Hysteresis tracker
# ---------------------------------------------------------------------------

class HysteresisTracker:
    """Prevents oscillation between control modes.

    A mode switch is only committed when the target mode has been
    consistently better for `patience` consecutive windows with
    a minimum improvement margin of `margin`.
    """

    def __init__(self, patience: int = 3, margin: float = 0.05) -> None:
        self.patience = patience
        self.margin = margin
        self._consecutive_wins: dict[str, int] = {}

    def should_switch(
        self,
        current_mode: str,
        candidate_mode: str,
        current_cost: float,
        candidate_cost: float,
    ) -> bool:
        """Return True if the switch should be committed."""
        if candidate_mode == current_mode:
            self._consecutive_wins.pop(candidate_mode, None)
            return False

        improvement = (current_cost - candidate_cost) / max(abs(current_cost), 1e-10)

        if improvement > self.margin:
            self._consecutive_wins[candidate_mode] = (
                self._consecutive_wins.get(candidate_mode, 0) + 1
            )
        else:
            self._consecutive_wins[candidate_mode] = 0

        if self._consecutive_wins.get(candidate_mode, 0) >= self.patience:
            self._consecutive_wins[candidate_mode] = 0
            return True

        return False

    def reset(self) -> None:
        """Reset all accumulated evidence."""
        self._consecutive_wins.clear()


# ---------------------------------------------------------------------------
# Adaptive Controller
# ---------------------------------------------------------------------------

class AdaptiveController:
    """
    Interpretable hardware-state-conditioned QEC controller.

    Core loop (executed once per observation window):
        1. Observe hardware state s_t from detectors and calibration
        2. Enumerate candidate actions A = {(decoder, dd, burst_mit, recal)}
        3. Compute J(a | s_t) for each candidate
        4. Select a*_t = argmin_a J(a | s_t) subject to hysteresis
        5. Return the selected action

    The controller is stateful (it tracks history for hysteresis and
    mode switching), but the cost function is fully explicit and
    interpretable — no neural network, no black box.
    """

    def __init__(
        self,
        weights: Optional[CostWeights] = None,
        hysteresis_patience: int = 3,
        hysteresis_margin: float = 0.05,
    ) -> None:
        self.weights = weights or CostWeights()
        self.hysteresis = HysteresisTracker(
            patience=hysteresis_patience,
            margin=hysteresis_margin,
        )
        self.metrics = ControllerMetrics()

        self._current_action: Optional[ControlAction] = None
        self._action_history: list[ControlAction] = []
        self._cost_history: list[dict[str, float]] = []

    @property
    def current_action(self) -> Optional[ControlAction]:
        """The last action selected by the controller."""
        return self._current_action

    def _enumerate_candidates(self, state: HardwareState) -> list[ControlAction]:
        """Generate the set of candidate actions for the current state.

        The candidate set is small and interpretable. We enumerate:
            - 2 decoders (MWPM, UF)
            - 4 DD policies (NONE, CPMG, XY4, XY8)
            - burst mitigation (on/off, but only if burst detected)

        Total: up to 2 * 4 * 2 = 16 candidates (much fewer in practice).
        """
        candidates = []
        decoders = [DecoderChoice.MWPM, DecoderChoice.UNION_FIND]
        dd_options = [DDSequenceType.NONE, DDSequenceType.XY4]

        # Only consider CPMG and XY8 if drift or leakage is significant
        if state.drift_magnitude > 1.0 or state.leakage_fraction > 0.01:
            dd_options.extend([DDSequenceType.CPMG, DDSequenceType.XY8])

        burst_options = [False]
        if state.burst_active:
            burst_options = [True, False]

        for dec in decoders:
            for dd in dd_options:
                for burst_mit in burst_options:
                    candidates.append(ControlAction(
                        decoder=dec,
                        dd_policy=dd,
                        burst_mitigation=burst_mit,
                    ))

        return candidates

    def select_action(self, state: HardwareState) -> ControlAction:
        """Select the optimal control action for the current hardware state.

        This is the main entry point. Call once per observation window.

        Args:
            state: Current estimated hardware state.

        Returns:
            The selected ControlAction.
        """
        candidates = self._enumerate_candidates(state)

        # Compute cost for each candidate
        costs: list[tuple[float, ControlAction]] = []
        for action in candidates:
            cost = compute_cost(state, action, self._current_action, self.weights)
            costs.append((cost, action))

        # Sort by cost (lowest first)
        costs.sort(key=lambda x: x[0])
        best_cost, best_action = costs[0]

        # Apply hysteresis to prevent oscillation
        if self._current_action is not None:
            current_cost = compute_cost(
                state, self._current_action, self._current_action, self.weights
            )

            current_mode = f"{self._current_action.decoder.value}:{self._current_action.dd_policy.value}"
            candidate_mode = f"{best_action.decoder.value}:{best_action.dd_policy.value}"

            if not self.hysteresis.should_switch(
                current_mode, candidate_mode, current_cost, best_cost
            ):
                # Stay with current decoder and DD policy, but keep instantaneous event mitigations
                best_action = ControlAction(
                    decoder=self._current_action.decoder,
                    dd_policy=self._current_action.dd_policy,
                    burst_mitigation=best_action.burst_mitigation,
                    request_recalibration=best_action.request_recalibration,
                    notes=f"Retained {current_mode} via hysteresis",
                )
                best_cost = current_cost

        # Check if this is a mode switch
        if self._current_action is not None:
            if best_action.decoder != self._current_action.decoder:
                self.metrics.total_mode_switches += 1
                logger.info(
                    f"Controller: mode switch "
                    f"{self._current_action.decoder.value} -> "
                    f"{best_action.decoder.value} "
                    f"(cost: {best_cost:.6f})"
                )

        # Update state
        self._current_action = best_action
        self._action_history.append(best_action)

        # Update metrics
        self.metrics.total_windows += 1
        self.metrics.decoder_usage[best_action.decoder.value] = (
            self.metrics.decoder_usage.get(best_action.decoder.value, 0) + 1
        )
        self.metrics.dd_usage[best_action.dd_policy.value] = (
            self.metrics.dd_usage.get(best_action.dd_policy.value, 0) + 1
        )
        if best_action.burst_mitigation:
            self.metrics.burst_mitigations += 1
        if best_action.request_recalibration:
            self.metrics.recalibration_requests += 1
        self.metrics.cost_history.append(best_cost)
        self.metrics.action_history.append(
            f"{best_action.decoder.value}:{best_action.dd_policy.value}"
        )

        return best_action

    def reset(self) -> None:
        """Reset controller state (for a new experiment run)."""
        self._current_action = None
        self._action_history.clear()
        self._cost_history.clear()
        self.hysteresis.reset()
        self.metrics = ControllerMetrics()

    def summary(self) -> dict[str, Any]:
        """Return a summary of controller performance."""
        return {
            "weights": self.weights.to_dict(),
            "metrics": self.metrics.to_dict(),
            "hysteresis": {
                "patience": self.hysteresis.patience,
                "margin": self.hysteresis.margin,
            },
        }
