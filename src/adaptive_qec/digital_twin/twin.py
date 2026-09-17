"""
Hardware digital twin.

Internal representation of the QPU:

    Qubit
     ├── T1
     ├── T2
     ├── readout error
     ├── 1Q fidelity
     ├── 2Q fidelity
     ├── leakage probability
     └── temporal behavior

Plus topology:
    q0 ─ q1 ─ q2
         │
         q3

The model predicts:
    P(logical failure) from the current estimated hardware state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from adaptive_qec.qpu.base import CalibrationSnapshot

logger = logging.getLogger(__name__)


@dataclass
class QubitState:
    """Current estimated state of a single qubit."""
    index: int
    t1_us: float = 0.0
    t2_us: float = 0.0
    readout_error: float = 0.0
    single_qubit_fidelity: float = 1.0
    two_qubit_fidelities: dict[int, float] = field(default_factory=dict)  # neighbor → fidelity
    leakage_probability: float = 0.0
    last_updated: str = ""

    # Temporal history
    t1_history: list[tuple[str, float]] = field(default_factory=list)
    t2_history: list[tuple[str, float]] = field(default_factory=list)
    readout_history: list[tuple[str, float]] = field(default_factory=list)


class HardwareDigitalTwin:
    """
    Maintains an internal model of the QPU hardware state.

    Updated from calibration snapshots and experimental observations.
    Predicts P(logical failure) from the current estimated state.
    """

    def __init__(self, num_qubits: int) -> None:
        self._num_qubits = num_qubits
        self._qubits: dict[int, QubitState] = {
            i: QubitState(index=i) for i in range(num_qubits)
        }
        self._topology: list[tuple[int, int]] = []
        self._update_count = 0

    def update_from_calibration(self, calibration: CalibrationSnapshot) -> None:
        """
        Update the digital twin from a calibration snapshot.

        Stores the new values and appends to temporal history.
        """
        timestamp = calibration.timestamp

        for qc in calibration.qubit_calibrations:
            idx = qc.qubit_index
            if idx not in self._qubits:
                self._qubits[idx] = QubitState(index=idx)

            state = self._qubits[idx]
            state.last_updated = timestamp

            if qc.t1_us is not None:
                state.t1_us = qc.t1_us
                state.t1_history.append((timestamp, qc.t1_us))
                # Keep last 100 entries
                if len(state.t1_history) > 100:
                    state.t1_history = state.t1_history[-100:]

            if qc.t2_us is not None:
                state.t2_us = qc.t2_us
                state.t2_history.append((timestamp, qc.t2_us))
                if len(state.t2_history) > 100:
                    state.t2_history = state.t2_history[-100:]

            if qc.readout_error is not None:
                state.readout_error = qc.readout_error
                state.readout_history.append((timestamp, qc.readout_error))
                if len(state.readout_history) > 100:
                    state.readout_history = state.readout_history[-100:]

            if qc.single_qubit_gate_error is not None:
                state.single_qubit_fidelity = 1.0 - qc.single_qubit_gate_error

        # Update gate fidelities
        for gc in calibration.gate_calibrations:
            if len(gc.qubits) == 2 and gc.error is not None:
                q1, q2 = gc.qubits
                if q1 in self._qubits:
                    self._qubits[q1].two_qubit_fidelities[q2] = 1.0 - gc.error
                if q2 in self._qubits:
                    self._qubits[q2].two_qubit_fidelities[q1] = 1.0 - gc.error

        self._topology = calibration.coupling_map
        self._update_count += 1
