"""
Adaptive scheduling experiment.

Tests the dynamic X/Z stabilizer scheduling claim:
    "Under biased noise (T1 ≠ T2), measuring the more-informative
     stabilizer type more frequently improves logical error rate
     compared to balanced 1:1 scheduling."

Experimental design:
    1. Generate surface code circuits at d=3
    2. Inject noise with controlled bias η = p_Z / p_X
    3. Compare stabilizer schedules:
       a. BALANCED: Equal X/Z frequency (1:1)
       b. X_HEAVY:  2:1 X:Z ratio (for dephasing-dominated noise)
       c. Z_HEAVY:  2:1 Z:X ratio (for relaxation-dominated noise)
       d. ADAPTIVE: Dynamic schedule selected by AdaptiveXZScheduler
    4. Sweep bias η from 0.1 to 10 to find the crossover
    5. Report improvement with Wilson score confidence intervals

Usage:
    python -m adaptive_qec.experiments.adaptive_scheduling
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

from adaptive_qec.qec.schedules import (
    ScheduleType,
    StabilizerSchedule,
    get_schedule,
    schedule_for_bias,
)
from adaptive_qec.qec.adaptive_scheduler import (
    AdaptiveSchedulerConfig,
    AdaptiveXZScheduler,
    DefectObservation,
)
from adaptive_qec.analysis.significance import wilson_score_ci

logger = logging.getLogger(__name__)


@dataclass
class SchedulingExperimentConfig:
    """Configuration for the adaptive scheduling experiment."""

    # QEC parameters
    code_distance: int = 3
    num_rounds: int = 4
    base_error_rate: float = 0.005

    # Bias sweep
    bias_values: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    )

    # Per-bias experiment
    shots_per_schedule: int = 50000
    num_windows: int = 50

    # Scheduler config
    ewma_alpha: float = 0.05
    theta_enter: float = 0.15
    theta_exit: float = 0.05

    # Output
    output_dir: str = "experiments/results/adaptive_scheduling"
    seed: int = 42


@dataclass
class BiasPointResult:
    """Result for one bias value and one schedule type."""
    bias_eta: float
    schedule_type: str
    total_errors: int
    total_shots: int
    ler: float
    ci_lower: float
    ci_upper: float
    adaptive_switches: int = 0


class AdaptiveSchedulingExperiment:
    """
    Bias sweep experiment for adaptive X/Z scheduling.

    Sweeps noise bias η from relaxation-dominated to dephasing-dominated,
    comparing fixed schedules against adaptive scheduling at each point.
    """

    def __init__(
        self,
        config: Optional[SchedulingExperimentConfig] = None,
    ) -> None:
        self._config = config or SchedulingExperimentConfig()
        self._rng = np.random.default_rng(self._config.seed)
        self._results: list[BiasPointResult] = []

    def run(self) -> list[BiasPointResult]:
        """Execute the full bias sweep experiment."""
        cfg = self._config
        t_start = time.time()

        logger.info(
            f"Starting adaptive scheduling experiment: "
            f"d={cfg.code_distance}, "
            f"bias sweep: {cfg.bias_values}"
        )

        schedules_to_test = [
            ScheduleType.BALANCED,
            ScheduleType.X_HEAVY,
            ScheduleType.Z_HEAVY,
