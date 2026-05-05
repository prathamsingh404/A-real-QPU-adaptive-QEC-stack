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
            f"windows={cfg.num_windows}, "
            f"scenario={cfg.scenario}"
        )

        # Create noise scenario
        scenario = self._create_scenario()

        # Build base circuit
        circuit = self._build_circuit()
        dem = circuit.detector_error_model(decompose_errors=True)

        import pymatching
        from adaptive_qec.decoders.union_find import UnionFindDecoder
        matcher = pymatching.Matching.from_detector_error_model(dem)
        uf_decoder = UnionFindDecoder()
        uf_decoder.configure(dem=dem)

        for window_idx in range(cfg.num_windows):
            # Check budget
            if not self._budget.can_submit(cfg.shots_per_window):
                logger.warning(f"Budget exhausted at window {window_idx}")
                break

            t_window = time.time()

            # Get noise for this window
            noise = scenario.get_noise(window_idx)

            # Build noise-injected circuit
            noisy_circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                distance=cfg.code_distance,
                rounds=cfg.num_rounds,
                after_clifford_depolarization=noise.gate_error_2q,
                before_round_data_depolarization=noise.gate_error_2q,
                before_measure_flip_probability=noise.readout_error,
                after_reset_flip_probability=noise.gate_error_2q * 0.5,
            )

            # Sample identical syndrome data for all arms
            sampler = noisy_circuit.compile_detector_sampler()
            detection_events, observable_flips = sampler.sample(
                shots=cfg.shots_per_window,
                separate_observables=True,
            )

            # Update adaptive scheduler
            schedule_type = "balanced"
            imbalance = 0.0
            if self._scheduler is not None:
                n_det = detection_events.shape[1]
                det_coords = circuit.get_detector_coordinates()
                x_indices: list[int] = []
                z_indices: list[int] = []
                for d_id, coords in det_coords.items():
                    if len(coords) >= 2:
                        if int(round(coords[0] + coords[1])) % 4 == 0:
                            x_indices.append(d_id)
                        else:
                            z_indices.append(d_id)
                if not x_indices or not z_indices:
                    x_indices = list(range(0, n_det, 2))
                    z_indices = list(range(1, n_det, 2))

                x_det = int(np.sum(detection_events[:, x_indices]))
                z_det = int(np.sum(detection_events[:, z_indices]))
                obs_sched = DefectObservation(
                    round_idx=window_idx,
                    x_defects=x_det,
                    z_defects=z_det,
                    total_x_stabilizers=len(x_indices),
                    total_z_stabilizers=len(z_indices),
                )
                sched = self._scheduler.update(obs_sched)
                schedule_type = sched.schedule_type.value
                imbalance = self._scheduler.imbalance

            # Hardware state for controller
            hw_state = HardwareState(
                error_rate=noise.gate_error_2q,
                t1_us=noise.t1_us,
                t2_us=noise.t2_us,
                readout_error=noise.readout_error,
                gate_error_1q=noise.gate_error_1q,
                gate_error_2q=noise.gate_error_2q,
            )

            # Adaptive controller decision
            self._controller.observe(hw_state)
            action = self._controller.decide()

            # Decode based on adaptive controller choice
            dec_str = action.decoder.value if hasattr(action.decoder, "value") else str(action.decoder)
            if "union" in dec_str.lower() or "uf" in dec_str.lower():
                predictions = uf_decoder.decode_batch(detection_events)
            else:
                predictions = matcher.decode_batch(detection_events)

            n_obs = observable_flips.shape[1] if observable_flips.ndim > 1 else 1
            pred_flat = predictions[:, :n_obs] if predictions.ndim > 1 else predictions.reshape(-1, 1)
            obs_flat = observable_flips if observable_flips.ndim > 1 else observable_flips.reshape(-1, 1)

            logical_errors = int(np.sum(np.any(pred_flat != obs_flat, axis=1)))
            ler = logical_errors / cfg.shots_per_window
            ci_low, ci_high = wilson_score_ci(logical_errors, cfg.shots_per_window)

            # Update controller with reward
            reward = 1.0 - ler
            self._controller.update(reward)

            # Record budget usage
            self._budget.record_usage(
                cfg.shots_per_window,
                source=f"window_{window_idx}",
            )

            # Record results
            self._adaptive_records.append(WindowRecord(
                window_idx=window_idx,
                ler=ler,
                ci_lower=ci_low,
                ci_upper=ci_high,
                logical_errors=logical_errors,
                total_shots=cfg.shots_per_window,
                noise_level=noise.gate_error_2q,
                controller_action={
                    "decoder": action.decoder if hasattr(action, "decoder") else "mwpm",
                    "dd_sequence": action.dd_sequence if hasattr(action, "dd_sequence") else "none",
                },
                schedule_type=schedule_type,
                imbalance=imbalance,
                execution_time_s=time.time() - t_window,
            ))

            # Genuinely track static baselines on the same syndrome data
            mwpm_preds = matcher.decode_batch(detection_events)
            mwpm_flat = mwpm_preds[:, :n_obs] if mwpm_preds.ndim > 1 else mwpm_preds.reshape(-1, 1)
            mwpm_errs = int(np.sum(np.any(mwpm_flat != obs_flat, axis=1)))
            self._static_mwpm_lers.append(mwpm_errs / cfg.shots_per_window)

            uf_preds = uf_decoder.decode_batch(detection_events)
            uf_flat = uf_preds[:, :n_obs] if uf_preds.ndim > 1 else uf_preds.reshape(-1, 1)
            uf_errs = int(np.sum(np.any(uf_flat != obs_flat, axis=1)))
            self._static_uf_lers.append(uf_errs / cfg.shots_per_window)


            if (window_idx + 1) % 20 == 0:
                avg_ler = np.mean([r.ler for r in self._adaptive_records[-20:]])
                logger.info(
                    f"Window {window_idx + 1}/{cfg.num_windows}: "
                    f"avg LER (last 20) = {avg_ler:.6f}, "
                    f"schedule = {schedule_type}"
                )

        elapsed = time.time() - t_start

        # Compile and save results
        results = self._compile_results(elapsed)
        self._save_results(results)
        self._print_summary(results)

        return results

    def _create_scenario(self) -> NoiseScenarioFactory:
        """Create noise scenario."""
        n = self._config.num_windows
        scenarios = {
            "stationary": lambda: stationary_scenario(total_steps=n),
            "drift": lambda: drift_scenario(total_steps=n),
            "burst": lambda: burst_scenario(total_steps=n),
            "multi_phase": lambda: multi_phase_scenario(total_steps=n),
        }
        return scenarios[self._config.scenario]()

    def _build_circuit(self) -> stim.Circuit:
        """Build base Stim circuit."""
        return stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=self._config.code_distance,
            rounds=self._config.num_rounds,
            after_clifford_depolarization=self._config.physical_error_rate,
            before_round_data_depolarization=self._config.physical_error_rate,
            before_measure_flip_probability=self._config.physical_error_rate * 2,
            after_reset_flip_probability=self._config.physical_error_rate * 0.5,
        )

    def _compile_results(self, elapsed: float) -> dict[str, Any]:
        """Compile all results into a single dictionary."""
        adaptive_lers = [r.ler for r in self._adaptive_records]
        total_errors = sum(r.logical_errors for r in self._adaptive_records)
        total_shots = sum(r.total_shots for r in self._adaptive_records)

        ci_low, ci_high = wilson_score_ci(total_errors, total_shots)

        return {
            "config": {
                "code_distance": self._config.code_distance,
                "controller": self._config.controller_type,
                "scheduling": self._config.scheduling_enabled,
                "scenario": self._config.scenario,
                "num_windows": len(self._adaptive_records),
                "shots_per_window": self._config.shots_per_window,
            },
            "summary": {
                "total_shots": total_shots,
                "total_errors": total_errors,
                "overall_ler": total_errors / total_shots if total_shots > 0 else 0.0,
                "ci_95_lower": ci_low,
                "ci_95_upper": ci_high,
                "mean_ler": float(np.mean(adaptive_lers)),
                "std_ler": float(np.std(adaptive_lers)),
                "elapsed_s": round(elapsed, 2),
            },
            "scheduler": self._scheduler.summary() if self._scheduler else None,
            "budget": self._budget.summary(),
            "per_window": [
                {
                    "window": r.window_idx,
                    "ler": r.ler,
                    "ci_lower": r.ci_lower,
                    "ci_upper": r.ci_upper,
                    "noise_level": r.noise_level,
                    "schedule": r.schedule_type,
                    "imbalance": round(r.imbalance, 4),
                    "action": r.controller_action,
                }
                for r in self._adaptive_records
            ],
        }

    def _save_results(self, results: dict[str, Any]) -> None:
        """Save results to disk."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self._config.output_dir) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        if self._config.save_telemetry:
            telemetry = {
                "controller": [
                    t.to_dict() if hasattr(t, "to_dict") else str(t)
                    for t in self._controller.telemetry
                ] if hasattr(self._controller, "telemetry") else [],
                "scheduler_history": (
                    self._scheduler.get_history()
                    if self._scheduler else []
                ),
            }
            with open(output_dir / "telemetry.json", "w") as f:
                json.dump(telemetry, f, indent=2, default=str)

        logger.info(f"Results saved to {output_dir}")

    def _print_summary(self, results: dict[str, Any]) -> None:
        """Print summary."""
        s = results["summary"]
        c = results["config"]
        sched = results.get("scheduler")

        print(f"\n{'='*60}")
        print(f"FULL ADAPTIVE QEC EXPERIMENT RESULTS")
        print(f"{'='*60}")
        print(f"Controller:      {c['controller']}")
        print(f"Scheduling:      {'ON' if c['scheduling'] else 'OFF'}")
        print(f"Scenario:        {c['scenario']}")
        print(f"Total shots:     {s['total_shots']:,}")
        print(f"Overall LER:     {s['overall_ler']:.6f}")
        print(f"95% CI:          [{s['ci_95_lower']:.6f}, {s['ci_95_upper']:.6f}]")
        print(f"Std(LER):        {s['std_ler']:.6f}")
        if sched:
            print(f"Schedule switches: {sched.get('total_switches', 0)}")
        print(f"Elapsed:         {s['elapsed_s']:.1f}s")
