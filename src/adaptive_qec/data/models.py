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
