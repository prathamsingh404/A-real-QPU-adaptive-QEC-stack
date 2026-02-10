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
