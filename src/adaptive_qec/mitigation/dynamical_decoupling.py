"""
Dynamical Decoupling (DD) integration for idle-time noise suppression.

When a qubit is idle during a QEC round (e.g. waiting for a two-qubit gate
on neighbors to finish, or during SWAP routing on heavy-hex), it accumulates
coherent ZZ crosstalk and non-Markovian low-frequency dephasing.

Indiscriminate DD can hurt: each DD pulse incurs gate error. An optimal
framework applies DD selectively — only when the dephasing noise prevented
exceeds the pulse error penalty.

Supported sequences:
    - CPMG: X - X (order 2, basic dephasing echo)
    - XY4: X - Y - X - Y (order 4, robust against pulse rotation errors)
    - XY8: XY4 + rotated XY4 (order 8, high fidelity)

Sources:
    - IBM Qiskit "Orbit" dynamical decoupling framework
    - Georgia Tech / ETH Zurich ADAPT framework (arXiv 2026)
    - Pokharel et al., "Demonstration of algorithmic quantum speedup for
      an abelian hidden subgroup problem with dynamical decoupling" (2023)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import stim

from adaptive_qec.digital_twin.twin import HardwareDigitalTwin

logger = logging.getLogger(__name__)


class DDSequenceType(str, Enum):
    """Dynamical decoupling pulse sequences."""
    NONE = "none"
    CPMG = "cpmg"  # 2 pulses: X - X
    XY4 = "xy4"    # 4 pulses: X - Y - X - Y
    XY8 = "xy8"    # 8 pulses: X - Y - X - Y - Y - X - Y - X


@dataclass
class DDSchedule:
    """Per-qubit dynamical decoupling schedule."""
    qubit_sequences: dict[int, DDSequenceType] = field(default_factory=dict)
    idle_windows_us: dict[int, float] = field(default_factory=dict)
    pulse_counts: dict[int, int] = field(default_factory=dict)
    protected_qubits: list[int] = field(default_factory=list)
    total_pulses_inserted: int = 0
    estimated_noise_reduction: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "qubit_sequences": {
                q: seq.value for q, seq in self.qubit_sequences.items()
            },
            "idle_windows_us": {
                q: round(t, 3) for q, t in self.idle_windows_us.items()
            },
            "protected_qubits": self.protected_qubits,
            "num_protected_qubits": len(self.protected_qubits),
            "total_pulses_inserted": self.total_pulses_inserted,
            "estimated_noise_reduction": round(self.estimated_noise_reduction, 4),
        }


class AdaptiveDDPlanner:
    """
    Decides and schedules dynamical decoupling pulses selectively
    based on digital twin hardware calibration.

    Decision rule:
        Apply sequence S on qubit q if:
            p_dephase(q, t_idle) > error_cost(S, q)
    """

    PULSE_COUNTS = {
        DDSequenceType.NONE: 0,
        DDSequenceType.CPMG: 2,
        DDSequenceType.XY4: 4,
        DDSequenceType.XY8: 8,
    }

    # Noise suppression factors (empirical from IBM Heron DD studies)
    SUPPRESSION_FACTORS = {
        DDSequenceType.NONE: 1.0,
        DDSequenceType.CPMG: 0.45,
        DDSequenceType.XY4: 0.22,
        DDSequenceType.XY8: 0.12,
    }

    def __init__(
        self,
        digital_twin: Optional[HardwareDigitalTwin] = None,
        single_qubit_pulse_error: float = 0.0003,
        default_sequence: DDSequenceType = DDSequenceType.XY4,
    ) -> None:
        """
        Args:
            digital_twin: Hardware model with T2 coherence times.
            single_qubit_pulse_error: Estimated error per DD pulse (1Q X/Y gate).
            default_sequence: Default candidate sequence when DD is beneficial.
        """
        self.twin = digital_twin or HardwareDigitalTwin(num_qubits=156)
        self.pulse_error = single_qubit_pulse_error
        self.default_sequence = default_sequence

    def plan_schedule(
        self,
        idle_map: dict[int, float],
        preferred_sequence: Optional[DDSequenceType] = None,
    ) -> DDSchedule:
        """
        Compute an optimal per-qubit DD schedule given idle durations.

        Args:
            idle_map: dict mapping qubit_index → idle_duration in microseconds.
            preferred_sequence: sequence type to use if DD is beneficial.

        Returns:
            DDSchedule specifying which qubits receive DD and their sequence.
        """
        seq_type = preferred_sequence or self.default_sequence
        num_pulses = self.PULSE_COUNTS[seq_type]
        pulse_cost = num_pulses * self.pulse_error

        schedule = DDSchedule(idle_windows_us=dict(idle_map))
        total_pulses = 0
        noise_savings = []

        for qubit, idle_us in idle_map.items():
            state = self.twin.get_qubit_state(qubit)
            t2 = state.t2_us if state and state.t2_us > 0 else 100.0  # default 100us if unknown

            # Dephasing probability during idle time: p = 1 - exp(-t / T2)
            p_dephase = 1.0 - float(np.exp(-idle_us / max(t2, 1.0)))

            # Decision rule: only apply DD if net noise improves
            suppressed_dephase = p_dephase * self.SUPPRESSION_FACTORS[seq_type]
            net_noise_with_dd = suppressed_dephase + pulse_cost

            if net_noise_with_dd < p_dephase:
                schedule.qubit_sequences[qubit] = seq_type
                schedule.pulse_counts[qubit] = num_pulses
                schedule.protected_qubits.append(qubit)
                total_pulses += num_pulses
                noise_savings.append(p_dephase - net_noise_with_dd)
            else:
                schedule.qubit_sequences[qubit] = DDSequenceType.NONE
                schedule.pulse_counts[qubit] = 0

        schedule.total_pulses_inserted = total_pulses
        schedule.estimated_noise_reduction = float(np.mean(noise_savings)) if noise_savings else 0.0

        logger.info(
            f"DD plan: {len(schedule.protected_qubits)}/{len(idle_map)} qubits protected, "
            f"total pulses={total_pulses}, avg saving={schedule.estimated_noise_reduction:.6f}"
        )
        return schedule

    @staticmethod
    def estimate_idle_map_from_circuit(
        circuit: stim.Circuit,
        gate_duration_us: float = 0.05,
    ) -> dict[int, float]:
        """
