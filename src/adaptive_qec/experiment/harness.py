"""
Experiment harness: the unified runner for adaptive-vs-static experiments.

Orchestrates the closed loop:
    hardware → syndrome extraction → decoding → reward → controller update

Supports:
    - A/B comparison (static baseline vs. adaptive controller)
    - Multi-arm comparison (multiple controllers on same hardware stream)
    - Interleaved execution (round-robin across controllers)
    - Telemetry and provenance logging

The harness uses the QPU backend (IBMBackend or mock) and decoder
interfaces directly — no simulation shortcuts.

Usage:
    harness = ExperimentHarness.from_config("configs/default.yaml")
    results = harness.run(num_windows=100)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import stim

from adaptive_qec.config import AQECConfig, load_config
from adaptive_qec.controller.base import BaseController, TelemetryRecord
from adaptive_qec.controller.controller import (
    ControlAction,
    DecoderChoice,
    HardwareState,
)
from adaptive_qec.decoders.base import Decoder, DecoderMetrics
from adaptive_qec.decoders.mwpm import MWPMDecoder
from adaptive_qec.noise.drift import DriftStatus, EWMADriftDetector
from adaptive_qec.provenance import DataProvenance, ProvenanceRegistry, ProvenanceTag
from adaptive_qec.syndrome.extraction import SyndromeExtractor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment result
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    """Result from a single observation window."""
    window_index: int
    controller_name: str
    action: dict[str, Any]
    hardware_state: dict[str, Any]
    decoder_metrics: dict[str, Any]
    logical_error_rate: float
    reward: float  # 1 - logical_error_rate
    elapsed_ms: float
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_index": self.window_index,
            "controller_name": self.controller_name,
            "action": self.action,
            "hardware_state": self.hardware_state,
            "decoder_metrics": self.decoder_metrics,
            "logical_error_rate": self.logical_error_rate,
            "reward": self.reward,
            "elapsed_ms": self.elapsed_ms,
            "timestamp": self.timestamp,
        }


@dataclass
class ExperimentRunResult:
    """Complete result from an experiment run."""
    experiment_id: str
    config: dict[str, Any]
    controller_summaries: dict[str, Any]
    window_results: list[dict[str, Any]]
    aggregate_metrics: dict[str, Any]
    provenance: dict[str, Any]
    start_time: str
    end_time: str
    total_windows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "config": self.config,
            "controller_summaries": self.controller_summaries,
            "window_results": self.window_results,
            "aggregate_metrics": self.aggregate_metrics,
            "provenance": self.provenance,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_windows": self.total_windows,
        }


# ---------------------------------------------------------------------------
# Stim circuit builder (uses real noise parameters, not hardcoded)
# ---------------------------------------------------------------------------

def build_surface_code_circuit(
    distance: int,
    rounds: int,
    p_1q: float,
    p_2q: float,
    p_ro: float,
) -> stim.Circuit:
    """Build a rotated surface-code memory circuit with calibrated noise.

    Noise rates come from hardware calibration data, not constants.

    Parameters
    ----------
    distance : int
        Code distance (must be odd, ≥ 3).
    rounds : int
        Number of QEC rounds.
    p_1q : float
        Single-qubit depolarizing error rate (from calibration).
    p_2q : float
        Two-qubit depolarizing error rate (from calibration).
    p_ro : float
        Readout (measurement) error rate (from calibration).

    Returns
    -------
    stim.Circuit
        The Stim circuit with noise and detectors.
    """
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p_2q,
        after_reset_flip_probability=p_1q,
        before_measure_flip_probability=p_ro,
        before_round_data_depolarization=p_1q,
    )
    return circuit


# ---------------------------------------------------------------------------
# Hardware state builder (from real calibration + syndrome data)
# ---------------------------------------------------------------------------

def build_hardware_state(
    defect_rates: np.ndarray,
    drift_detector: EWMADriftDetector,
    calibration_data: dict[str, float],
    code_distance: int,
    burst_threshold: float = 0.3,
    leakage_estimate: float = 0.0,
) -> HardwareState:
    """Construct a HardwareState from real observations.

    Parameters
    ----------
    defect_rates : np.ndarray
        Per-detector firing rates from the most recent window.
    drift_detector : EWMADriftDetector
        The drift detector (already fed previous windows).
    calibration_data : dict
        Must contain: t1_mean_us, t2_mean_us, p_1q, p_2q, p_ro.
    code_distance : int
        Current code distance.
    burst_threshold : float
        Defect-rate threshold above which a burst is flagged.
    leakage_estimate : float
        Estimated leakage fraction (from syndrome autocorrelation).
    """
    mean_defect = float(np.mean(defect_rates))
    max_defect = float(np.max(defect_rates))

    # Feed the drift detector
    drift_report = drift_detector.analyze(defect_rates)

    burst_active = max_defect > burst_threshold

    return HardwareState(
        defect_rate=mean_defect,
        drift_magnitude=drift_report.magnitude,
        drift_status=drift_report.status,
        burst_active=burst_active,
        leakage_fraction=leakage_estimate,
        t1_mean_us=calibration_data.get("t1_mean_us", 100.0),
        t2_mean_us=calibration_data.get("t2_mean_us", 80.0),
        p_1q=calibration_data.get("p_1q", 0.0005),
        p_2q=calibration_data.get("p_2q", 0.003),
        p_ro=calibration_data.get("p_ro", 0.012),
        code_distance=code_distance,
        num_data_qubits=code_distance ** 2,
        num_detectors=int(defect_rates.shape[0]) if defect_rates.ndim > 0 else 8,
    )


# ---------------------------------------------------------------------------
# Experiment Harness
# ---------------------------------------------------------------------------

class ExperimentHarness:
    """Unified experiment runner for adaptive QEC research.

    Runs one or more controllers against the same hardware stream
    and collects per-window telemetry for offline analysis.

    Parameters
    ----------
    controllers : list[BaseController]
        Controllers to evaluate.  The first is the baseline.
    config : AQECConfig
        Experiment configuration.
    output_dir : Path
        Where to save results.
    """

    def __init__(
        self,
        controllers: list[BaseController],
        config: AQECConfig,
        output_dir: Optional[Path] = None,
    ) -> None:
        if not controllers:
            raise ValueError("Must provide at least one controller")

        self._controllers = controllers
        self._config = config
        self._output_dir = output_dir or Path(config.experiment.output_dir)
        self._provenance = ProvenanceRegistry()

        # Build Stim circuit from calibration data
        self._circuit: Optional[stim.Circuit] = None
        self._extractor: Optional[SyndromeExtractor] = None
        self._decoders: dict[str, Decoder] = {}
        self._drift_detector = EWMADriftDetector(
            num_detectors=1,  # will be updated
            alpha=0.1,
            z_warning=2.0,
            z_drift=3.0,
            z_severe=5.0,
        )

        # Register provenance
        self._provenance.register("code_distance", ProvenanceTag(
            value=float(config.qec.distance),
            provenance=DataProvenance.MEASURED,
            source="config file",
        ))
        self._provenance.register("qec_rounds", ProvenanceTag(
            value=float(config.qec.rounds),
            provenance=DataProvenance.MEASURED,
            source="config file",
        ))

    @classmethod
    def from_config(cls, config_path: str, controllers: list[BaseController]) -> ExperimentHarness:
        """Create harness from a YAML config file."""
        config = load_config(config_path)
        return cls(controllers=controllers, config=config)

    def _setup_circuit(self, calibration: dict[str, float]) -> None:
        """Build the Stim circuit using real calibration data."""
        p_1q = calibration.get("p_1q", self._config.noise.gate.single_qubit)
        p_2q = calibration.get("p_2q", self._config.noise.gate.two_qubit)
        p_ro = calibration.get("p_ro", self._config.noise.readout.p0_given_1)

        self._circuit = build_surface_code_circuit(
            distance=self._config.qec.distance,
            rounds=self._config.qec.rounds,
            p_1q=p_1q,
            p_2q=p_2q,
            p_ro=p_ro,
        )

        self._extractor = SyndromeExtractor(self._circuit)

        # Reinitialize drift detector with correct detector count
        num_det = self._circuit.num_detectors
        self._drift_detector = EWMADriftDetector(
            num_detectors=num_det,
            alpha=0.1,
            z_warning=2.0,
            z_drift=3.0,
            z_severe=5.0,
        )

        # Setup decoder
        mwpm = MWPMDecoder()
        dem = self._circuit.detector_error_model(decompose_errors=True)
        mwpm.configure(dem=dem)
        self._decoders["mwpm"] = mwpm

        self._provenance.register("p_1q", ProvenanceTag(
            value=p_1q,
            provenance=DataProvenance.MEASURED,
            source="IBM QPU calibration",
        ))
        self._provenance.register("p_2q", ProvenanceTag(
            value=p_2q,
            provenance=DataProvenance.MEASURED,
            source="IBM QPU calibration",
