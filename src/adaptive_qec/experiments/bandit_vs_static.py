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


# -----------------------------------------------------------------------
# Arm result tracking
# -----------------------------------------------------------------------

@dataclass
class ArmWindowResult:
    """Result for one arm in one evaluation window."""
    window_idx: int
    arm_name: str
    logical_errors: int
    total_shots: int
    ler: float
    ci_lower: float
    ci_upper: float
    controller_action: Optional[dict[str, Any]] = None
    execution_time_s: float = 0.0


@dataclass
class ExperimentSummary:
    """Executive summary of the experiment."""
    total_windows: int
    total_shots_per_arm: int
    arms: dict[str, dict[str, Any]] = field(default_factory=dict)
    best_static: str = ""
    best_adaptive: str = ""
    adaptive_improvement_pct: float = 0.0
    p_value: float = 1.0
    significant: bool = False


# -----------------------------------------------------------------------
# Main experiment runner
# -----------------------------------------------------------------------

class BanditVsStaticExperiment:
    """
    Full bandit vs static comparative experiment.

    Generates syndrome data under non-stationary noise, then runs
    multiple controller strategies on identical data to compare
    logical error rates with rigorous statistical analysis.
    """

    def __init__(self, config: Optional[BanditExperimentConfig] = None) -> None:
        self._config = config or BanditExperimentConfig()
        self._rng = np.random.default_rng(self._config.seed)

        # Build arm set
        self._arms = build_arm_set()

        # Initialize controllers
        self._controllers = self._init_controllers()

        # Results tracking
        self._results: dict[str, list[ArmWindowResult]] = {
            name: [] for name in self._controllers
        }
        self._noise_scenario: Optional[NoiseScenarioFactory] = None

    def _init_controllers(self) -> dict[str, Any]:
        """Initialize all controller variants."""
        cfg = self._config
        arms = self._arms

        controllers = {
            "static_mwpm": StaticController(
                decoder="mwpm", dd_sequence="none", schedule="balanced"
            ),
            "static_uf_xy4": StaticController(
                decoder="union_find", dd_sequence="xy4", schedule="balanced"
            ),
            "exp3": Exp3Controller(
                arms=arms, gamma=cfg.exp3_gamma,
            ),
            "dase": DASEController(
                arms=arms,
                exploration_bonus=cfg.dase_exploration_bonus,
                drift_window=cfg.dase_drift_window,
            ),
            "sprt": SPRTController(
                arms=arms,
                alpha=cfg.sprt_alpha,
                beta=cfg.sprt_beta,
                p0=cfg.sprt_p0,
                p1=cfg.sprt_p1,
            ),
        }
        return controllers

    def _create_scenario(self) -> NoiseScenarioFactory:
        """Create the noise scenario based on configuration."""
        total_steps = self._config.num_windows
        scenario_map = {
            "stationary": lambda: stationary_scenario(total_steps=total_steps),
            "drift": lambda: drift_scenario(total_steps=total_steps),
            "burst": lambda: burst_scenario(total_steps=total_steps),
            "multi_phase": lambda: multi_phase_scenario(total_steps=total_steps),
        }
        factory_fn = scenario_map.get(self._config.scenario)
        if factory_fn is None:
            raise ValueError(f"Unknown scenario: {self._config.scenario}")
        return factory_fn()

    def run(self) -> ExperimentSummary:
        """
        Execute the full experiment.

        Returns:
            ExperimentSummary with comparative results.
        """
        cfg = self._config
        t_start = time.time()

        logger.info(
            f"Starting BanditVsStatic experiment: "
            f"d={cfg.code_distance}, "
            f"windows={cfg.num_windows}, "
            f"shots_per_window={cfg.shots_per_window}, "
            f"scenario={cfg.scenario}"
        )

        # Create noise scenario
        self._noise_scenario = self._create_scenario()

        # Generate Stim circuit for baseline
        circuit = self._build_stim_circuit()
        dem = circuit.detector_error_model(decompose_errors=True)

        # Build decoders
        import pymatching
        from adaptive_qec.decoders.union_find import UnionFindDecoder
        mwpm_matcher = pymatching.Matching.from_detector_error_model(dem)
        uf_decoder = UnionFindDecoder()
        uf_decoder.configure_from_dem(dem)

        # Main experiment loop
        for window_idx in range(cfg.num_windows):
            noise = self._noise_scenario.get_noise(window_idx)

            # Build noise-injected circuit for this window
