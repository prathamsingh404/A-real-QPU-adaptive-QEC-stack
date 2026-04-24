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
