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
    burst_detected: bool = False     # Alias for burst_active
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

        if self.burst_detected and not self.burst_active:
            object.__setattr__(self, "burst_active", True)
        elif self.burst_active and not self.burst_detected:
            object.__setattr__(self, "burst_detected", True)

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


class InterventionType(str, Enum):
    """Categorization of adaptive actions to separate classical vs physical effects."""
    CLASSICAL_ONLY = "classical_only"
    PHYSICAL_CIRCUIT = "physical_circuit"
    HYBRID = "hybrid"


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

    @property
    def is_classical_only(self) -> bool:
        """Return True if this action only involves classical decoder adaptation."""
        dd_is_none = (
            self.dd_policy == DDSequenceType.NONE
            or (hasattr(self.dd_policy, "value") and self.dd_policy.value == "none")
            or str(self.dd_policy).lower() in ("none", "ddsequencetype.none")
        )
        return dd_is_none and not self.request_recalibration

    @property
    def is_physical(self) -> bool:
        """Return True if this action alters physical quantum circuit execution."""
        return not self.is_classical_only

    @property
    def intervention_type(self) -> InterventionType:
        if self.is_classical_only:
            return InterventionType.CLASSICAL_ONLY
        elif self.decoder is not None:
            return InterventionType.HYBRID
        else:
            return InterventionType.PHYSICAL_CIRCUIT

    @property
    def dd_pattern(self) -> Any:
        return self.dd_policy

    @property
    def reason(self) -> str:
        return self.notes

    @property
    def recalibrate_dem(self) -> bool:
        return self.request_recalibration


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
