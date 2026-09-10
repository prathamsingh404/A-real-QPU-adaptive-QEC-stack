"""
Data models for experiment records and calibration snapshots.

Every QPU experiment saves a complete record that can be reproduced
six months later. This is the data contract between the experiment
pipeline and the storage layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np


@dataclass
class ExperimentRecord:
    """
    Complete record of a single QPU experiment.

    Every field listed in the spec is captured:
    - experiment_id, timestamp, backend, qubit_mapping, circuit,
      shots, measurement_results, calibration_snapshot,
      software_version, compiler_configuration
    """
    experiment_id: str
    timestamp: str
    backend: str
    provider: str
    qubit_mapping: Optional[dict[int, int]]
    circuit_qasm: str
    shots: int
    num_qubits_measured: int
    measurement_outcomes: np.ndarray   # (shots, num_measured_qubits)
    counts: dict[str, int]
    calibration_snapshot: dict[str, Any]
    software_versions: dict[str, str]
    compiler_config: dict[str, Any]
    config_snapshot: dict[str, Any]
    git_commit: Optional[str] = None
    execution_time_s: Optional[float] = None
    job_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create_timestamp() -> str:
        """Create ISO 8601 timestamp in UTC."""
        return datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d = {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "backend": self.backend,
            "provider": self.provider,
            "qubit_mapping": self.qubit_mapping,
            "circuit_qasm": self.circuit_qasm,
            "shots": self.shots,
            "num_qubits_measured": self.num_qubits_measured,
            "counts": self.counts,
            "calibration_snapshot": self.calibration_snapshot,
            "software_versions": self.software_versions,
            "compiler_config": self.compiler_config,
            "config_snapshot": self.config_snapshot,
            "git_commit": self.git_commit,
            "execution_time_s": self.execution_time_s,
            "job_id": self.job_id,
            "metadata": self.metadata,
        }
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentRecord:
        """Reconstruct from dictionary. measurement_outcomes loaded separately."""
        return cls(
            experiment_id=data["experiment_id"],
            timestamp=data["timestamp"],
            backend=data["backend"],
            provider=data["provider"],
            qubit_mapping=data.get("qubit_mapping"),
            circuit_qasm=data["circuit_qasm"],
            shots=data["shots"],
            num_qubits_measured=data["num_qubits_measured"],
            measurement_outcomes=np.array([]),  # loaded from .npy
            counts=data["counts"],
            calibration_snapshot=data["calibration_snapshot"],
            software_versions=data["software_versions"],
            compiler_config=data["compiler_config"],
            config_snapshot=data["config_snapshot"],
