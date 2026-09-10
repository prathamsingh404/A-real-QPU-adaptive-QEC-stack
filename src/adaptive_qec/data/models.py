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
            git_commit=data.get("git_commit"),
            execution_time_s=data.get("execution_time_s"),
            job_id=data.get("job_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DetectorRecord:
    """Detector extraction results for an experiment."""
    experiment_id: str
    num_rounds: int
    num_detectors: int
    syndrome_tensor: np.ndarray       # (shots, rounds * detectors) or (shots, R, Nd)
    observable_flips: np.ndarray      # (shots, num_observables)
    detector_coordinates: Optional[np.ndarray] = None  # (num_detectors, ndim)

    def to_dict(self) -> dict[str, Any]:
        """Metadata only — arrays saved separately."""
        return {
            "experiment_id": self.experiment_id,
            "num_rounds": self.num_rounds,
            "num_detectors": self.num_detectors,
            "syndrome_shape": list(self.syndrome_tensor.shape),
            "observable_shape": list(self.observable_flips.shape),
        }


@dataclass
class DecoderRecord:
    """Decoder results for an experiment."""
    experiment_id: str
    decoder_name: str
    num_shots: int
    num_logical_errors: int
    logical_error_rate: float
    corrections: np.ndarray            # (shots, num_observables)
    decode_time_s: float
    per_shot_latency_us: Optional[np.ndarray] = None  # (shots,)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "decoder_name": self.decoder_name,
            "num_shots": self.num_shots,
            "num_logical_errors": self.num_logical_errors,
            "logical_error_rate": self.logical_error_rate,
            "decode_time_s": self.decode_time_s,
            "metrics": self.metrics,
        }


@dataclass
class ExperimentMetrics:
    """Aggregated metrics for an experiment."""
    experiment_id: str
    logical_error_rate: float
    logical_error_rate_ci_low: float
    logical_error_rate_ci_high: float
    confidence_level: float
    physical_error_rate: Optional[float] = None
    lambda_ratio: Optional[float] = None   # logical / physical
    per_round_error_rate: Optional[float] = None
    decoder_latency_mean_us: Optional[float] = None
    decoder_latency_p99_us: Optional[float] = None
    decoder_throughput: Optional[float] = None  # syndromes / second
    total_pipeline_time_s: Optional[float] = None
    noise_statistics: dict[str, Any] = field(default_factory=dict)
    drift_report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "logical_error_rate": self.logical_error_rate,
            "logical_error_rate_ci": [
                self.logical_error_rate_ci_low,
                self.logical_error_rate_ci_high,
            ],
            "confidence_level": self.confidence_level,
            "physical_error_rate": self.physical_error_rate,
            "lambda_ratio": self.lambda_ratio,
            "per_round_error_rate": self.per_round_error_rate,
            "decoder_latency_mean_us": self.decoder_latency_mean_us,
            "decoder_latency_p99_us": self.decoder_latency_p99_us,
            "decoder_throughput": self.decoder_throughput,
            "total_pipeline_time_s": self.total_pipeline_time_s,
            "noise_statistics": self.noise_statistics,
            "drift_report": self.drift_report,
        }
