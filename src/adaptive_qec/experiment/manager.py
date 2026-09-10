"""
Experiment manager — orchestrates the full QEC experiment pipeline.

Pipeline:
    1. Load config
    2. Initialize QPU backend
    3. Capture calibration snapshot
    4. Generate QEC circuit (Stim)
    5. Convert circuit for QPU
    6. Execute on real QPU
    7. Extract syndromes from raw measurements
    8. Run decoder
    9. Compute noise statistics + drift analysis
    10. Compute metrics with confidence intervals
    11. Save everything for reproducibility
    12. Generate plots
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from adaptive_qec.analysis.metrics import compute_logical_error_metrics
from adaptive_qec.analysis.statistics import compute_confidence_interval
from adaptive_qec.config import AdaptiveQECConfig
from adaptive_qec.data.models import (
    DecoderRecord,
    ExperimentMetrics,
    ExperimentRecord,
)
from adaptive_qec.data.store import ExperimentStore
from adaptive_qec.decoders.registry import get_decoder
from adaptive_qec.experiment.reproducibility import capture_reproducibility_info
from adaptive_qec.noise.characterization import NoiseCharacterizer
from adaptive_qec.noise.drift import CompositeDriftDetector, DriftReport
from adaptive_qec.qec.circuits import CircuitGenerator
from adaptive_qec.qpu.base import CalibrationSnapshot, QPUBackend
from adaptive_qec.qpu.registry import get_backend
from adaptive_qec.runtime.profiler import PipelineProfiler
from adaptive_qec.syndrome.extraction import SyndromeExtractor

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    Orchestrates end-to-end QEC experiments on real QPU hardware.

    Every experiment creates a complete, reproducible record.
    """

    def __init__(self, config: AdaptiveQECConfig) -> None:
        self._config = config
        self._store = ExperimentStore(config.experiment.output_dir)
        self._drift_detector = CompositeDriftDetector(warmup_samples=5)
        self._noise_characterizer = NoiseCharacterizer()
        self._profiler = PipelineProfiler()
        self._qpu: Optional[QPUBackend] = None

    def connect_qpu(self) -> None:
        """Connect to the configured QPU backend."""
        self._qpu = get_backend(self._config)
        self._qpu.connect()
        logger.info(f"Connected to QPU: {self._config.hardware.backend}")

    def run_experiment(
        self,
        experiment_name: Optional[str] = None,
        shots: Optional[int] = None,
    ) -> ExperimentMetrics:
        """
        Run a complete QEC experiment end-to-end.

        Args:
            experiment_name: Optional custom name for the experiment.
            shots: Override shot count from config.

        Returns:
            ExperimentMetrics with full results.
        """
        if self._qpu is None:
            raise RuntimeError("QPU not connected. Call connect_qpu() first.")

        profiler = PipelineProfiler()
        experiment_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        actual_shots = shots or self._config.experiment.shots

        logger.info(
            f"Starting experiment {experiment_id}: "
            f"code={self._config.qec.code.value}, "
            f"d={self._config.qec.distance}, "
            f"R={self._config.qec.rounds}, "
            f"shots={actual_shots}, "
            f"backend={self._config.hardware.backend}"
        )

        # ---- Step 1: Capture calibration ----
        profiler.start_stage("calibration")
        calibration = self._qpu.get_calibration()
        profiler.end_stage("calibration")
        logger.info("Calibration snapshot captured")

        # ---- Step 2: Generate QEC circuit ----
        profiler.start_stage("circuit_generation")
        circuit_gen = CircuitGenerator(self._config.qec, self._config.noise)
        stim_circuit = circuit_gen.generate_stim_circuit()
        qiskit_circuit = circuit_gen.stim_to_qiskit(stim_circuit)
