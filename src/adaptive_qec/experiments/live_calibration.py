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
