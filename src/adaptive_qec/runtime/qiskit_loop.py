"""
Qiskit Runtime closed-loop coordinator.

Implements the batched closed-loop interface with real IBM Heron hardware
via Qiskit Runtime Sessions and SamplerV2. This is NOT simulation —
it connects to actual IBM quantum processors.

Architecture:
    1. Open a Qiskit Runtime Session (holds QPU reservation)
    2. Submit batched QEC circuits (N_batch shots per iteration)
    3. Retrieve syndrome bitstrings
    4. Feed to controller (bandit + SPRT) and DEM calibrator
    5. Update circuits/decoder weights based on controller decision
    6. Submit next batch within the active session
    7. Repeat until shot budget exhausted or convergence detected

The session-based approach minimizes queue wait time between batches,
enabling real-time closed-loop adaptation on hardware.

References:
    - Qiskit Runtime Sessions documentation
    - IBM Quantum "Orbit" dynamical decoupling framework
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

@dataclass
class RuntimeLoopConfig:
    """
    Configuration for the Qiskit Runtime closed-loop coordinator.

    Attributes:
        backend_name: IBM backend to target (e.g. 'ibm_marrakesh').
        shots_per_batch: Number of shots per circuit submission.
        max_batches: Maximum number of batches before stopping.
        max_total_shots: Hard cap on total shots consumed.
        session_timeout_s: Session timeout in seconds.
        warmup_batches: Number of batches for forced exploration.
        convergence_threshold: Regret improvement threshold for
            early stopping.
        convergence_window: Number of batches to check for convergence.
        output_dir: Directory to save experiment artifacts.
        dry_run: If True, use a fake backend instead of real hardware.
    """

    backend_name: str = "ibm_marrakesh"
    shots_per_batch: int = 1000
    max_batches: int = 100
    max_total_shots: int = 200_000
    session_timeout_s: int = 7200  # 2 hours
    warmup_batches: int = 5
    convergence_threshold: float = 0.001
    convergence_window: int = 10
    output_dir: str = "experiments/hardware_runs"
    dry_run: bool = False

    def validate(self) -> None:
        """Validate configuration."""
        if self.shots_per_batch < 100:
            raise ValueError(
                f"shots_per_batch must be >= 100, got {self.shots_per_batch}"
            )
        if self.max_batches < 1:
            raise ValueError(
                f"max_batches must be >= 1, got {self.max_batches}"
            )
        if self.max_total_shots < self.shots_per_batch:
            raise ValueError(
                f"max_total_shots ({self.max_total_shots}) must be >= "
                f"shots_per_batch ({self.shots_per_batch})"
            )


# -----------------------------------------------------------------------
# Batch result
# -----------------------------------------------------------------------

@dataclass
class BatchResult:
    """
    Result from a single batch execution on hardware.

    Attributes:
        batch_idx: Batch sequence number.
        shots: Number of shots executed.
        syndromes: Raw syndrome bitstrings (shots × num_detectors).
        observables: Logical observable outcomes (shots × num_observables).
        logical_errors: Number of logical errors detected.
        logical_error_rate: LER for this batch.
        controller_action: The action taken by the controller.
        calibration_snapshot: Hardware calibration at time of execution.
        execution_time_s: Wall-clock time for this batch.
        timestamp: UTC timestamp of batch completion.
        job_id: IBM Quantum job ID.
    """

    batch_idx: int
    shots: int
    syndromes: Optional[np.ndarray] = None
    observables: Optional[np.ndarray] = None
    logical_errors: int = 0
    logical_error_rate: float = 0.0
    controller_action: Optional[dict[str, Any]] = None
    calibration_snapshot: Optional[dict[str, Any]] = None
    execution_time_s: float = 0.0
    timestamp: str = ""
    job_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (without large arrays)."""
        return {
            "batch_idx": self.batch_idx,
            "shots": self.shots,
            "logical_errors": self.logical_errors,
            "logical_error_rate": self.logical_error_rate,
            "controller_action": self.controller_action,
            "execution_time_s": round(self.execution_time_s, 3),
            "timestamp": self.timestamp,
            "job_id": self.job_id,
        }


# -----------------------------------------------------------------------
# Runtime loop
# -----------------------------------------------------------------------

class QiskitRuntimeLoop:
    """
    Closed-loop batched Qiskit Runtime session driver.

    Orchestrates the full adaptive QEC feedback loop on real hardware:
    circuit submission → syndrome retrieval → controller update →
    circuit/decoder adjustment → next submission.

    This class handles:
    - Session management (open, keepalive, close)
    - Batched circuit transpilation and submission
    - Syndrome extraction from SamplerV2 results
    - Controller and DEM calibrator integration
    - Shot budget enforcement
    - Provenance-tagged artifact serialization
    """

    def __init__(
        self,
        config: RuntimeLoopConfig,
        controller: Any = None,
        dem_calibrator: Any = None,
        scheduler: Any = None,
        budget_manager: Any = None,
    ) -> None:
        config.validate()
        self._config = config
        self._controller = controller
        self._dem_calibrator = dem_calibrator
        self._scheduler = scheduler
        self._budget_manager = budget_manager

        # Session state
        self._session: Any = None
        self._sampler: Any = None
        self._backend: Any = None

        # Execution state
        self._run_id = str(uuid.uuid4())[:8]
        self._batch_results: list[BatchResult] = []
        self._total_shots: int = 0
        self._total_errors: int = 0
        self._is_running: bool = False
        self._start_time: Optional[float] = None

    @property
    def run_id(self) -> str:
        """Unique identifier for this experiment run."""
        return self._run_id

    @property
    def total_shots(self) -> int:
        """Total shots executed so far."""
        return self._total_shots

    @property
    def total_batches(self) -> int:
        """Number of batches completed."""
        return len(self._batch_results)

    @property
    def overall_ler(self) -> float:
        """Overall logical error rate across all batches."""
        if self._total_shots == 0:
            return 0.0
        return self._total_errors / self._total_shots

    def connect(self) -> None:
        """
        Establish connection to IBM Quantum backend.

        Opens a Qiskit Runtime Session for the configured backend.
        In dry_run mode, uses a fake backend for testing.
        """
        if self._config.dry_run:
            logger.info(
                "DRY RUN mode: using local simulator instead of "
                f"{self._config.backend_name}"
            )
            self._backend = self._create_fake_backend()
