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
class CalibrationHistory:
    """
    Time-series container tracking the temporal drift of calibration parameters.

    Stores chronological CalibrationSnapshots and provides trajectory extraction
    for T1, T2, readout error, and gate errors with drift metrics and summary reports.
    """
    snapshots: list[CalibrationSnapshot] = field(default_factory=list)

    def add_snapshot(self, snapshot: CalibrationSnapshot) -> None:
        """Append a new calibration snapshot to the history."""
        self.snapshots.append(snapshot)

    def __len__(self) -> int:
        return len(self.snapshots)

    @property
    def timestamps(self) -> list[str]:
        """List of snapshot timestamps."""
        return [s.timestamp for s in self.snapshots]

    def get_t1_trajectory(self, qubit_index: int) -> np.ndarray:
        """Extract T1 time-series for a specific physical qubit."""
        values = []
        for s in self.snapshots:
            qc = s.get_qubit(qubit_index)
            values.append(qc.t1_us if (qc and qc.t1_us is not None) else np.nan)
        return np.array(values, dtype=np.float64)

    def get_t2_trajectory(self, qubit_index: int) -> np.ndarray:
        """Extract T2 time-series for a specific physical qubit."""
        values = []
        for s in self.snapshots:
            qc = s.get_qubit(qubit_index)
            values.append(qc.t2_us if (qc and qc.t2_us is not None) else np.nan)
        return np.array(values, dtype=np.float64)

    def get_readout_error_trajectory(self, qubit_index: int) -> np.ndarray:
        """Extract readout error time-series for a specific physical qubit."""
        values = []
        for s in self.snapshots:
            qc = s.get_qubit(qubit_index)
            values.append(qc.readout_error if (qc and qc.readout_error is not None) else np.nan)
        return np.array(values, dtype=np.float64)

    def get_gate_error_trajectory(self, qubits: tuple[int, ...], gate_name: Optional[str] = None) -> np.ndarray:
        """Extract gate error time-series across snapshots for specified qubits."""
        values = []
        for s in self.snapshots:
            err = np.nan
            for gc in s.gate_calibrations:
                if gc.qubits == qubits and (gate_name is None or gc.gate_name == gate_name):
                    err = gc.error if gc.error is not None else np.nan
                    break
            values.append(err)
        return np.array(values, dtype=np.float64)

    def compute_drift_statistics(self, qubit_index: int) -> dict[str, Any]:
        """Compute mean, std, relative coefficient of variation (CV), min, and max for a qubit."""
        t1 = self.get_t1_trajectory(qubit_index)
        t2 = self.get_t2_trajectory(qubit_index)
        ro = self.get_readout_error_trajectory(qubit_index)

        def _stats(arr: np.ndarray) -> dict[str, float]:
            valid = arr[np.isfinite(arr)]
            if len(valid) == 0:
                return {"mean": 0.0, "std": 0.0, "cv": 0.0, "min": 0.0, "max": 0.0}
            mean = float(np.mean(valid))
            std = float(np.std(valid))
            cv = float(std / mean) if mean > 0 else 0.0
            return {"mean": mean, "std": std, "cv": cv, "min": float(np.min(valid)), "max": float(np.max(valid))}

        return {
            "qubit_index": qubit_index,
            "num_snapshots": len(self.snapshots),
            "t1": _stats(t1),
            "t2": _stats(t2),
            "readout_error": _stats(ro),
        }

    def summary(self) -> dict[str, Any]:
        """Return aggregate summary across all tracked snapshots and qubits."""
        if not self.snapshots:
            return {"total_snapshots": 0, "qubits_tracked": 0, "backend_name": "unknown"}

        first_snap = self.snapshots[0]
        qubits = [qc.qubit_index for qc in first_snap.qubit_calibrations]
        return {
            "total_snapshots": len(self.snapshots),
            "start_time": self.snapshots[0].timestamp,
            "end_time": self.snapshots[-1].timestamp,
            "backend_name": first_snap.backend_name,
            "qubits_tracked": len(qubits),
        }


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

    All research code interacts with the QPU through this interface.
    Swap providers without rewriting experiment code.
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the backend."""
        ...

    @abstractmethod
    def run(
        self,
        circuit: Any,
        shots: int,
        qubit_mapping: Optional[dict[int, int]] = None,
    ) -> ExperimentResult:
        """
        Execute a circuit on the QPU.

        Args:
            circuit: Quantum circuit to execute.
            shots: Number of measurement shots.
            qubit_mapping: Logical → physical qubit mapping.

        Returns:
            ExperimentResult with raw measurement outcomes.
        """
        ...

    @abstractmethod
    def get_calibration(self) -> CalibrationSnapshot:
        """
        Capture current calibration snapshot.

        Returns T1, T2, readout errors, gate errors, coupling map —
        everything the provider exposes.
        """
        ...

    @abstractmethod
    def get_topology(self) -> TopologyInfo:
        """Get qubit connectivity information."""
        ...

    @abstractmethod
    def get_backend_info(self) -> BackendInfo:
        """Get static backend information."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is currently available."""
        ...

    def get_results(self, job_id: str) -> ExperimentResult:
        """
        Retrieve results from a previously submitted job.

        Not all backends support async job retrieval. Default raises.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support async job retrieval"
        )
