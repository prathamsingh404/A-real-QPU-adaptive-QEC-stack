"""
Abstract QPU backend interface.

This is the non-negotiable abstraction that decouples research code
from any single provider's API. All QPU interactions go through this
interface.

    qpu.run(circuit, shots)
    qpu.get_calibration()
    qpu.get_topology()
    qpu.get_backend_info()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class BackendInfo:
    """Static information about a QPU backend."""
    name: str
    provider: str
    num_qubits: int
    topology_type: str
    version: str
    status: str  # "online" | "offline" | "maintenance"
    max_shots: int
    max_circuits: int
    basis_gates: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologyInfo:
    """Qubit connectivity information."""
    num_qubits: int
    coupling_map: list[tuple[int, int]]
    adjacency: dict[int, list[int]] = field(default_factory=dict)

    def neighbors(self, qubit: int) -> list[int]:
        """Get neighboring qubits."""
        if not self.adjacency:
            self._build_adjacency()
        return self.adjacency.get(qubit, [])

    def _build_adjacency(self) -> None:
        """Build adjacency dict from coupling map."""
        self.adjacency = {}
        for q1, q2 in self.coupling_map:
            self.adjacency.setdefault(q1, []).append(q2)
            self.adjacency.setdefault(q2, []).append(q1)


@dataclass
class QubitCalibration:
    """Calibration data for a single qubit."""
    qubit_index: int
    t1_us: Optional[float] = None            # T1 in microseconds
    t2_us: Optional[float] = None            # T2 in microseconds
    readout_error: Optional[float] = None    # readout assignment error
    readout_length_ns: Optional[float] = None
    single_qubit_gate_error: Optional[float] = None
    frequency_ghz: Optional[float] = None
    anharmonicity_ghz: Optional[float] = None


@dataclass
class GateCalibration:
    """Calibration data for a gate on specific qubits."""
    gate_name: str
    qubits: tuple[int, ...]
    error: Optional[float] = None
    gate_length_ns: Optional[float] = None


@dataclass
class CalibrationSnapshot:
    """
    Complete calibration snapshot at a point in time.

    Captures: T1, T2, readout errors, gate errors, coupling map —
    everything the provider exposes. This allows investigation of
    hardware_state(t) rather than treating each QPU execution as isolated.
    """
    timestamp: str
    backend_name: str
    qubit_calibrations: list[QubitCalibration] = field(default_factory=list)
    gate_calibrations: list[GateCalibration] = field(default_factory=list)
    coupling_map: list[tuple[int, int]] = field(default_factory=list)
    additional_properties: dict[str, Any] = field(default_factory=dict)

    def get_qubit(self, index: int) -> Optional[QubitCalibration]:
        """Get calibration for a specific qubit."""
        for qc in self.qubit_calibrations:
            if qc.qubit_index == index:
                return qc
        return None

    def readout_errors(self) -> np.ndarray:
        """Get readout errors as an array."""
        errors = []
        for qc in sorted(self.qubit_calibrations, key=lambda q: q.qubit_index):
            errors.append(qc.readout_error if qc.readout_error is not None else 0.0)
        return np.array(errors)

    def t1_values(self) -> np.ndarray:
        """Get T1 values as an array (microseconds)."""
        values = []
        for qc in sorted(self.qubit_calibrations, key=lambda q: q.qubit_index):
            values.append(qc.t1_us if qc.t1_us is not None else 0.0)
        return np.array(values)

    def t2_values(self) -> np.ndarray:
        """Get T2 values as an array (microseconds)."""
        values = []
        for qc in sorted(self.qubit_calibrations, key=lambda q: q.qubit_index):
            values.append(qc.t2_us if qc.t2_us is not None else 0.0)
        return np.array(values)


@dataclass
class ExperimentResult:
    """
    Result from a QPU experiment execution.

    Contains raw measurement outcomes for every shot, plus metadata.
    """
    experiment_id: str
    backend_name: str
    shots: int
    measurement_outcomes: np.ndarray  # shape: (shots, num_measured_qubits)
    counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_time_s: Optional[float] = None
    job_id: Optional[str] = None


class QPUBackend(ABC):
    """
    Abstract QPU backend interface.

