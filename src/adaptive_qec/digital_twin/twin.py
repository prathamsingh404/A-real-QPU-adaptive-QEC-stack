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

        logger.info(
            f"Digital twin updated (update #{self._update_count}): "
            f"{len(calibration.qubit_calibrations)} qubits, "
            f"{len(calibration.gate_calibrations)} gates"
        )

    def predict_logical_failure_rate(
        self,
        data_qubits: list[int],
        ancilla_qubits: list[int],
        code_distance: int,
    ) -> float:
        """
        Predict P(logical failure) from the current hardware state.

        Uses a simple model based on average error rates and code distance.
        More sophisticated models can be added in V1+.

        Rough model: p_L ≈ A * (p/p_th)^(d+1)/2
        where p is the average physical error rate.
        """
        # Collect error rates for relevant qubits
        all_qubits = data_qubits + ancilla_qubits
        errors = []
        for q in all_qubits:
            if q in self._qubits:
                state = self._qubits[q]
                # Use readout error + gate error as approximate physical error
                err = state.readout_error + (1 - state.single_qubit_fidelity)
                # Add 2Q gate errors
                for neighbor, fidelity in state.two_qubit_fidelities.items():
                    if neighbor in all_qubits:
                        err += (1 - fidelity)
                errors.append(err)

        if not errors:
            return 0.5  # no data

        avg_error = np.mean(errors)
        # Simple threshold model with p_th ≈ 1%
        p_th = 0.01
        if avg_error < p_th:
            p_logical = 0.1 * (avg_error / p_th) ** ((code_distance + 1) / 2)
        else:
            p_logical = min(0.5, avg_error * code_distance)

        return float(p_logical)

    def get_qubit_state(self, qubit: int) -> Optional[QubitState]:
        """Get the current state of a qubit."""
        return self._qubits.get(qubit)

    def get_worst_qubits(self, metric: str = "readout_error", n: int = 10) -> list[int]:
        """
        Find the N worst-performing qubits by a given metric.

        Useful for qubit selection and calibration targeting.
        """
        values = []
        for idx, state in self._qubits.items():
            val = getattr(state, metric, 0.0)
            values.append((idx, val))

        values.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in values[:n]]

    def summary(self) -> dict[str, Any]:
        """Get a summary of the current hardware state."""
        t1s = [s.t1_us for s in self._qubits.values() if s.t1_us > 0]
        t2s = [s.t2_us for s in self._qubits.values() if s.t2_us > 0]
        readouts = [s.readout_error for s in self._qubits.values() if s.readout_error > 0]

        return {
            "num_qubits": self._num_qubits,
            "update_count": self._update_count,
            "t1_mean_us": float(np.mean(t1s)) if t1s else 0.0,
            "t2_mean_us": float(np.mean(t2s)) if t2s else 0.0,
            "readout_error_mean": float(np.mean(readouts)) if readouts else 0.0,
            "num_edges": len(self._topology),
        }
