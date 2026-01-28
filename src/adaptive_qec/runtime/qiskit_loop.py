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
