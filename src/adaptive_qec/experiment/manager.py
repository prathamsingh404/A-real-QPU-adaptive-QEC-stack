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
        measurement_mapping = circuit_gen.get_measurement_mapping(stim_circuit)
        profiler.end_stage("circuit_generation")
        logger.info(f"Circuit generated: {stim_circuit.num_qubits} qubits")

        # ---- Step 3: Execute on QPU ----
        profiler.start_stage("qpu_execution")
        qpu_result = self._qpu.run(qiskit_circuit, shots=actual_shots)
        profiler.end_stage("qpu_execution")
        logger.info(
            f"QPU execution complete: {qpu_result.shots} shots, "
            f"{len(qpu_result.counts)} unique outcomes"
        )

        # ---- Step 4: Extract syndromes ----
        profiler.start_stage("syndrome_extraction")
        extractor = SyndromeExtractor(stim_circuit)
        detection_events, observable_flips = extractor.extract_from_measurements(
            qpu_result.measurement_outcomes
        )
        detector_record = extractor.create_detector_record(
            experiment_id=experiment_id,
            detection_events=detection_events,
            observable_flips=observable_flips,
            num_rounds=self._config.qec.rounds,
        )
        profiler.end_stage("syndrome_extraction")
        logger.info(
            f"Syndrome extraction: {detection_events.shape[1]} detectors, "
            f"{observable_flips.shape[1]} observables"
        )

        # ---- Step 5: Decode ----
        profiler.start_stage("decoding")
        decoder = get_decoder(self._config.decoder.baseline.value)
        decoder.configure(circuit=stim_circuit)
        decoder_metrics = decoder.decode_batch(detection_events, observable_flips)
        profiler.end_stage("decoding")
        logger.info(
            f"Decoding complete: LER={decoder_metrics.logical_error_rate:.6f}, "
            f"P99={decoder_metrics.latency_p99_us:.1f}μs"
        )

        # ---- Step 6: Noise characterization ----
        profiler.start_stage("noise_analysis")
        noise_profile = self._noise_characterizer.characterize(
            detection_events=detection_events,
            num_rounds=self._config.qec.rounds,
            calibration=calibration,
        )
        profiler.end_stage("noise_analysis")

        # ---- Step 7: Drift detection ----
        profiler.start_stage("drift_detection")
        from adaptive_qec.noise.statistics import compute_detector_statistics
        det_stats = compute_detector_statistics(detection_events)
        drift_report = self._drift_detector.update(det_stats.detection_rates)
        profiler.end_stage("drift_detection")
        if drift_report.status.value != "stable":
            logger.warning(f"DRIFT: {drift_report.status.value}, magnitude={drift_report.magnitude:.2f}")

        # ---- Step 8: Statistical analysis ----
        profiler.start_stage("analysis")
        ci_low, ci_high = compute_confidence_interval(
            n_errors=decoder_metrics.num_logical_errors,
            n_total=decoder_metrics.total_shots,
            confidence=self._config.analysis.confidence_level,
        )
        profiler.end_stage("analysis")

        # ---- Step 9: Build experiment metrics ----
        pipeline_timing = profiler.get_report()
        metrics = ExperimentMetrics(
            experiment_id=experiment_id,
            logical_error_rate=decoder_metrics.logical_error_rate,
            logical_error_rate_ci_low=ci_low,
            logical_error_rate_ci_high=ci_high,
            confidence_level=self._config.analysis.confidence_level,
            physical_error_rate=noise_profile.estimated_physical_error_rate,
            decoder_latency_mean_us=decoder_metrics.latency_mean_us,
            decoder_latency_p99_us=decoder_metrics.latency_p99_us,
            decoder_throughput=decoder_metrics.throughput_shots_per_s,
            total_pipeline_time_s=pipeline_timing.get("total_time_s", 0),
            noise_statistics=noise_profile.to_dict(),
            drift_report=drift_report.to_dict(),
        )

        # ---- Step 10: Save everything ----
        profiler.start_stage("saving")
        repro_info = capture_reproducibility_info()

        # Serialize calibration for storage
        calibration_dict = {
            "timestamp": calibration.timestamp,
            "backend": calibration.backend_name,
            "qubit_count": len(calibration.qubit_calibrations),
            "t1_mean_us": float(calibration.t1_values().mean()),
            "t2_mean_us": float(calibration.t2_values().mean()),
            "readout_error_mean": float(calibration.readout_errors().mean()),
        }

        experiment_record = ExperimentRecord(
            experiment_id=experiment_id,
            timestamp=ExperimentRecord.create_timestamp(),
            backend=self._config.hardware.backend,
            provider=self._config.hardware.provider.value,
            qubit_mapping=None,
            circuit_qasm=str(qiskit_circuit) if qiskit_circuit else "",
            shots=actual_shots,
            num_qubits_measured=qpu_result.measurement_outcomes.shape[1],
            measurement_outcomes=qpu_result.measurement_outcomes,
            counts=qpu_result.counts,
            calibration_snapshot=calibration_dict,
            software_versions=repro_info["software_versions"],
            compiler_config={"optimization_level": 1},
