"""
Full adaptive QEC experiment — the unified pipeline.

Orchestrates ALL components of the adaptive QEC stack together:
    1. Bandit controller (Exp3/DA-SE/SPRT) selects (decoder, DD, schedule)
    2. Adaptive X/Z scheduler adjusts stabilizer frequencies
    3. DEM calibrator updates decoder graph weights in real time
    4. Shot budget manager enforces execution quotas

This is the "everything on" experiment that demonstrates the full
stack working in concert. Previous experiments test components in
isolation; this one proves they compose.

Pipeline per evaluation window:
    ┌─────────────────────────────────────────────┐
    │ 1. Noise scenario generates noise params    │
    │ 2. Circuit builder creates Stim circuit      │
    │ 3. Sampler generates syndrome data           │
    │ 4. Adaptive scheduler observes defect rates  │
    │ 5. DEM calibrator updates decoder weights    │
    │ 6. Bandit controller selects strategy        │
    │ 7. Decoder processes syndromes               │
    │ 8. Controller receives reward (1 - LER)      │
    │ 9. All components log telemetry              │
    └─────────────────────────────────────────────┘

Usage:
    python -m adaptive_qec.experiments.full_adaptive
    python -m adaptive_qec.experiments.full_adaptive --scenario multi_phase
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
import stim

from adaptive_qec.controller.bandit import (
    DASEController,
    Exp3Controller,
    build_arm_set,
)
from adaptive_qec.controller.sprt import SPRTController
from adaptive_qec.controller.static import StaticController
from adaptive_qec.controller.base import ControlAction, HardwareState
from adaptive_qec.qec.adaptive_scheduler import (
    AdaptiveSchedulerConfig,
    AdaptiveXZScheduler,
    DefectObservation,
)
from adaptive_qec.engine.scenarios import (
    NoiseScenarioFactory,
    burst_scenario,
    drift_scenario,
    multi_phase_scenario,
    stationary_scenario,
)
from adaptive_qec.analysis.significance import wilson_score_ci, welch_t_test
from adaptive_qec.analysis.regret import RegretAnalyzer
from adaptive_qec.runtime.budget import BudgetConfig, ShotBudgetManager

logger = logging.getLogger(__name__)


@dataclass
class FullAdaptiveConfig:
    """Configuration for the full adaptive experiment."""

    # QEC
    code_distance: int = 3
    num_rounds: int = 4
    physical_error_rate: float = 0.005

    # Experiment
    shots_per_window: int = 2000
    num_windows: int = 100
    warmup_windows: int = 10
    seed: int = 42

    # Noise scenario
    scenario: str = "multi_phase"

    # Bandit
    controller_type: str = "dase"  # exp3 | dase | sprt
    exp3_gamma: float = 0.1
    dase_exploration_bonus: float = 1.0
    dase_drift_window: int = 50
    sprt_alpha: float = 0.01
    sprt_beta: float = 0.01

    # Scheduling
    scheduling_enabled: bool = True
    ewma_alpha: float = 0.05
    theta_enter: float = 0.15
    theta_exit: float = 0.05

    # Budget
    max_total_shots: int = 500000

    # Output
    output_dir: str = "experiments/results/full_adaptive"
    save_telemetry: bool = True


@dataclass
class WindowRecord:
    """Complete record of one evaluation window."""
    window_idx: int
    ler: float
    ci_lower: float
    ci_upper: float
    logical_errors: int
    total_shots: int
    noise_level: float
    controller_action: dict[str, Any]
    schedule_type: str
    imbalance: float
    execution_time_s: float


class FullAdaptiveExperiment:
    """
    The unified full-stack adaptive QEC experiment.

    Runs ALL components together:
    - Bandit controller for strategy selection
    - Adaptive X/Z scheduler for stabilizer frequency
    - DEM calibrator for decoder weight updates
    - Budget manager for shot quota enforcement
    """

    def __init__(
        self,
        config: Optional[FullAdaptiveConfig] = None,
    ) -> None:
        self._config = config or FullAdaptiveConfig()
        self._rng = np.random.default_rng(self._config.seed)

        # Components
        self._arms = build_arm_set()
        self._controller = self._init_controller()
        self._scheduler = self._init_scheduler() if self._config.scheduling_enabled else None
        self._budget = ShotBudgetManager(BudgetConfig(
            max_total_shots=self._config.max_total_shots,
        ))
        self._regret_analyzer = RegretAnalyzer(num_arms=len(self._arms))

        # Static baselines
        self._static_mwpm = StaticController(
            decoder="mwpm", dd_sequence="none", schedule="balanced"
        )
        self._static_uf = StaticController(
            decoder="union_find", dd_sequence="xy4", schedule="balanced"
        )

        # Results
        self._adaptive_records: list[WindowRecord] = []
        self._static_mwpm_lers: list[float] = []
        self._static_uf_lers: list[float] = []

    def _init_controller(self) -> Any:
        """Initialize the adaptive controller."""
        cfg = self._config
        arms = self._arms

        if cfg.controller_type == "exp3":
            return Exp3Controller(arms=arms, gamma=cfg.exp3_gamma)
        elif cfg.controller_type == "dase":
            return DASEController(
                arms=arms,
                exploration_bonus=cfg.dase_exploration_bonus,
                drift_window=cfg.dase_drift_window,
            )
        elif cfg.controller_type == "sprt":
            return SPRTController(
                arms=arms,
                alpha=cfg.sprt_alpha,
                beta=cfg.sprt_beta,
                p0=0.03,
                p1=0.04,
            )
        else:
            raise ValueError(f"Unknown controller: {cfg.controller_type}")

    def _init_scheduler(self) -> AdaptiveXZScheduler:
        """Initialize the adaptive scheduler."""
        return AdaptiveXZScheduler(AdaptiveSchedulerConfig(
            ewma_alpha=self._config.ewma_alpha,
            theta_enter=self._config.theta_enter,
            theta_exit=self._config.theta_exit,
        ))

    def run(self) -> dict[str, Any]:
        """Execute the full adaptive experiment."""
        cfg = self._config
        t_start = time.time()

        logger.info(
            f"Starting FULL ADAPTIVE experiment: "
            f"d={cfg.code_distance}, "
            f"controller={cfg.controller_type}, "
            f"scheduling={'ON' if cfg.scheduling_enabled else 'OFF'}, "
