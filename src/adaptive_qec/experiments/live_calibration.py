"""
Live DEM Calibration Experiment.

Validates Phase 4 / Stage 3 of the Adaptive QEC Stack:
Evaluates real-time closed-loop Detector Error Model (DEM) graph reweighting
under drifting physical noise.

Compares:
    1. Static Factory MWPM (fixed matching graph from nominal calibration)
    2. Live-Calibrated MWPM (dynamic DEM reweighting from sliding-window syndrome/hardware telemetry)
    3. Static Union-Find Baseline

Usage:
    python -m adaptive_qec.experiments.live_calibration
    python -m adaptive_qec.experiments.live_calibration --windows 30 --shots 1000
    python -m adaptive_qec.experiments.live_calibration --scenario drift --distance 3
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pymatching
import stim

from adaptive_qec.analysis.significance import (
    StatisticalTestResult,
    welch_t_test,
    wilson_score_ci,
)
from adaptive_qec.decoders.dem_calibration import CalibratedDEM, DEMCalibrator
from adaptive_qec.decoders.union_find import UnionFindDecoder
from adaptive_qec.engine.scenarios import (
    NoiseScenarioFactory,
    NoiseSnapshot,
    burst_scenario,
    drift_scenario,
    multi_phase_scenario,
    stationary_scenario,
)

logger = logging.getLogger(__name__)


@dataclass
class LiveCalibrationConfig:
    """Configuration for live DEM calibration experiment."""

    code_distance: int = 3
    num_rounds: int = 3
    num_windows: int = 40
    shots_per_window: int = 1000
    scenario: str = "drift"
    nominal_error_rate: float = 0.003
    smoothing: float = 0.35
    seed: int = 42
    output_dir: str = "experiments/results/live_calibration"


@dataclass
class WindowComparison:
    """Per-window evaluation record."""

    window_idx: int
    p_2q: float
    p_ro: float
    static_errors: int
    calibrated_errors: int
    uf_errors: int
    total_shots: int
    static_ler: float
    calibrated_ler: float
    uf_ler: float


class LiveCalibrationExperiment:
    """Experiment comparing static vs live-calibrated decoders under drift."""

    def __init__(self, config: LiveCalibrationConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.seed)

        # Baseline nominal circuit and static factory matcher
        self._nominal_circuit = self._build_circuit(
            p_2q=config.nominal_error_rate,
            p_ro=config.nominal_error_rate * 2.0,
        )
        self._static_dem = self._nominal_circuit.detector_error_model(decompose_errors=True)
        self._static_matcher = pymatching.Matching.from_detector_error_model(self._static_dem)

        # DEM Calibrator initialized with nominal circuit
        self._calibrator = DEMCalibrator(
            circuit=self._nominal_circuit,
            min_probability=1e-6,
            max_probability=0.4999,
        )

        # Union-Find decoder
        self._uf_decoder = UnionFindDecoder()
        self._uf_decoder.configure_from_dem(self._static_dem)

        # Noise scenario
        self._scenario = self._create_scenario()

        # Telemetry
        self._records: list[WindowComparison] = []

    def _build_circuit(self, p_2q: float, p_ro: float) -> stim.Circuit:
        """Construct surface code circuit with specified physical noise rates."""
        d = self._config.code_distance
        r = self._config.num_rounds
        return stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=d,
            rounds=r,
            after_clifford_depolarization=p_2q,
            before_round_data_depolarization=p_2q,
            before_measure_flip_probability=p_ro,
            after_reset_flip_probability=p_ro * 0.5,
        )

    def _create_scenario(self) -> NoiseScenarioFactory:
        """Create noise scenario based on config."""
        n = self._config.num_windows
        scenarios = {
            "stationary": lambda: stationary_scenario(total_steps=n),
            "drift": lambda: drift_scenario(total_steps=n, rate=0.0003),
            "burst": lambda: burst_scenario(total_steps=n),
            "multi_phase": lambda: multi_phase_scenario(total_steps=n),
        }
        factory_fn = scenarios.get(self._config.scenario, scenarios["drift"])
        return factory_fn()

    def run(self) -> dict[str, Any]:
        """Execute the live calibration experiment."""
        logger.info(
            f"Starting LIVE CALIBRATION experiment: d={self._config.code_distance}, "
            f"windows={self._config.num_windows}, scenario={self._config.scenario}"
        )
        t_start = time.time()
        cfg = self._config

        for window_idx in range(cfg.num_windows):
            snapshot: NoiseSnapshot = self._scenario.step()
            p_2q = snapshot.p_2q
            p_ro = snapshot.p_ro

            # Generate real drifting circuit and sample syndromes
            noisy_circuit = self._build_circuit(p_2q=p_2q, p_ro=p_ro)
            sampler = noisy_circuit.compile_detector_sampler(seed=int(self._rng.integers(0, 2**31)))
            detection_events, observables = sampler.sample(
                shots=cfg.shots_per_window,
                separate_observables=True,
            )

            obs_flat = observables[:, :1] if observables.ndim > 1 else observables.reshape(-1, 1)

            # 1. Decode with Static Factory MWPM
            static_preds = self._static_matcher.decode_batch(detection_events)
            static_flat = static_preds[:, :1] if static_preds.ndim > 1 else static_preds.reshape(-1, 1)
            static_errs = int(np.sum(np.any(static_flat != obs_flat, axis=1)))

            # 2. Decode with Live-Calibrated MWPM
            calibrated_dem: CalibratedDEM = self._calibrator.calibrate_from_syndromes(
                detection_events,
                smoothing=cfg.smoothing,
                calibration_timestamp=f"window_{window_idx}",
            )
            calibrated_matcher = calibrated_dem.to_matching()
            cal_preds = calibrated_matcher.decode_batch(detection_events)
            cal_flat = cal_preds[:, :1] if cal_preds.ndim > 1 else cal_preds.reshape(-1, 1)
            cal_errs = int(np.sum(np.any(cal_flat != obs_flat, axis=1)))

            # 3. Decode with Static Union-Find
            uf_preds = self._uf_decoder.decode_batch(detection_events)
            uf_flat = uf_preds[:, :1] if uf_preds.ndim > 1 else uf_preds.reshape(-1, 1)
