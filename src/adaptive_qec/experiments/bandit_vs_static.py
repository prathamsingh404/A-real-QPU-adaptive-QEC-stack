"""
Bandit vs Static experiment.

Compares the adaptive bandit controller (Exp3/DA-SE/SPRT) against
fixed-policy baselines under non-stationary noise scenarios.

This experiment answers the paper's central claim:
    "An online bandit controller that adaptively selects
     (decoder, DD policy) outperforms any single fixed strategy
     under non-stationary noise."

Experimental design:
    1. Generate a non-stationary noise scenario
       (drift + bursts + phase transitions)
    2. For each evaluation window of N shots:
       a. STATIC-MWPM:       Fixed MWPM, no DD
       b. STATIC-UF-XY4:     Fixed UF, fixed XY4 DD
       c. ADAPTIVE-EXP3:     Exp3 bandit selects (decoder, DD)
       d. ADAPTIVE-DASE:     DA-SE bandit selects (decoder, DD)
       e. ADAPTIVE-SPRT:     SPRT-gated bandit
    3. All arms decode THE SAME syndrome data (eliminates sampling noise)
    4. Wilson score 95% CI on per-window LER
    5. Benjamini-Hochberg FDR correction for multiple comparisons

Output:
    - experiments/results/bandit_vs_static/<timestamp>/
        ├── results.json          Full results with per-window data
        ├── summary.json          Executive summary
        ├── ler_over_time.png     LER comparison plot
        ├── cumulative_regret.png Regret tracking
        └── arm_selection.png     Controller arm selection timeline

Usage:
    python -m adaptive_qec.experiments.bandit_vs_static
    python -m adaptive_qec.experiments.bandit_vs_static --config configs/adaptive.yaml
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

from adaptive_qec.controller.base import ControlAction, HardwareState
from adaptive_qec.controller.bandit import (
    DASEController,
    Exp3Controller,
    build_arm_set,
)
from adaptive_qec.controller.sprt import SPRTController
from adaptive_qec.controller.static import StaticController
from adaptive_qec.engine.scenarios import (
    NoiseScenarioFactory,
    burst_scenario,
    drift_scenario,
    multi_phase_scenario,
    stationary_scenario,
)
from adaptive_qec.analysis.significance import (
    StatisticalTestResult,
    welch_t_test,
    wilson_score_ci,
    benjamini_hochberg,
)
from adaptive_qec.analysis.regret import RegretAnalyzer

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------

@dataclass
class BanditExperimentConfig:
    """Configuration for the bandit vs static experiment."""

    # QEC parameters
    code_distance: int = 3
    num_rounds: int = 4
    physical_error_rate: float = 0.005

    # Experiment structure
    shots_per_window: int = 2000
    num_windows: int = 50
    warmup_windows: int = 5
    seed: int = 42

    # Noise scenario
    scenario: str = "multi_phase"  # stationary | drift | burst | multi_phase

    # Bandit parameters
    exp3_gamma: float = 0.1
    dase_exploration_bonus: float = 1.0
    dase_drift_window: int = 50
    sprt_alpha: float = 0.01
    sprt_beta: float = 0.01
    sprt_p0: float = 0.03
    sprt_p1: float = 0.04

    # Output
    output_dir: str = "experiments/results/bandit_vs_static"
    save_syndromes: bool = False
    save_telemetry: bool = True

