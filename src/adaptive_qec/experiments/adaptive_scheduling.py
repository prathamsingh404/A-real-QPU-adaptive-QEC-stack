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
        ]

        for eta in cfg.bias_values:
            logger.info(f"\n--- Bias η = {eta:.2f} ---")

            # Build biased circuit
            p_x, p_z = self._bias_to_error_rates(
                cfg.base_error_rate, eta
            )
            logger.info(f"  p_X = {p_x:.6f}, p_Z = {p_z:.6f}")

            # Fixed schedules
            for sched_type in schedules_to_test:
                result = self._run_fixed_schedule(
                    eta=eta, p_x=p_x, p_z=p_z, schedule_type=sched_type
                )
                self._results.append(result)
                logger.info(
                    f"  {sched_type.value}: LER = {result.ler:.6f} "
                    f"[{result.ci_lower:.6f}, {result.ci_upper:.6f}]"
                )

            # Adaptive schedule
            result = self._run_adaptive_schedule(eta=eta, p_x=p_x, p_z=p_z)
            self._results.append(result)
            logger.info(
                f"  adaptive: LER = {result.ler:.6f} "
                f"[{result.ci_lower:.6f}, {result.ci_upper:.6f}] "
                f"({result.adaptive_switches} switches)"
            )

        elapsed = time.time() - t_start
        logger.info(f"Experiment complete in {elapsed:.1f}s")

        self._save_results()
        return self._results

    def _bias_to_error_rates(
        self,
        p_total: float,
        eta: float,
    ) -> tuple[float, float]:
        """
        Convert total error rate and bias η to p_X and p_Z.

        Under biased noise model:
            p_Z = η · p_X
            p_total = p_X + p_Z + p_Y ≈ p_X + p_Z (ignoring p_Y)
            p_X = p_total / (1 + η)
            p_Z = η · p_total / (1 + η)
        """
        p_x = p_total / (1 + eta)
        p_z = eta * p_total / (1 + eta)
        return p_x, p_z

    def _build_biased_circuit(
        self,
        p_x: float,
        p_z: float,
        schedule_type: Optional[ScheduleType] = None,
    ) -> stim.Circuit:
        """Build a Stim circuit with biased noise and stabilizer schedule."""
        d = self._config.code_distance
        r = self._config.num_rounds

        # Model the physical effect of check schedule frequency on error accumulation:
        # Measuring an observable more frequently shortens idle accumulation time Delta t.
        # X-heavy schedule measures X-checks twice as often, halving accumulation time for Z errors,
        # but doubling accumulation time for X errors on Z checks.
        scale_x = 1.0
        scale_z = 1.0
        if schedule_type == ScheduleType.X_HEAVY:
            scale_z = 0.55
            scale_x = 1.80
        elif schedule_type == ScheduleType.Z_HEAVY:
            scale_x = 0.55
            scale_z = 1.80
        elif schedule_type == ScheduleType.EXTREME_X:
            scale_z = 0.35
            scale_x = 2.50
        elif schedule_type == ScheduleType.EXTREME_Z:
            scale_x = 0.35
            scale_z = 2.50

        eff_px = min(p_x * scale_x, 0.20)
        eff_pz = min(p_z * scale_z, 0.20)
        p_total = eff_px + eff_pz

        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=d,
            rounds=r,
            after_clifford_depolarization=p_total,
            before_round_data_depolarization=p_total,
            before_measure_flip_probability=min(p_total * 1.5, 0.20),
            after_reset_flip_probability=min(p_total * 0.5, 0.20),
        )
        return circuit

    def _run_fixed_schedule(
        self,
        eta: float,
        p_x: float,
        p_z: float,
        schedule_type: ScheduleType,
    ) -> BiasPointResult:
        """Run experiment with a fixed stabilizer schedule."""
        circuit = self._build_biased_circuit(p_x, p_z, schedule_type=schedule_type)
        sampler = circuit.compile_detector_sampler()
        dem = circuit.detector_error_model(decompose_errors=True)

        import pymatching
        matcher = pymatching.Matching.from_detector_error_model(dem)

        shots = self._config.shots_per_schedule
        detection_events, observable_flips = sampler.sample(
            shots=shots,
            separate_observables=True,
        )

        predictions = matcher.decode_batch(detection_events)
        n_obs = observable_flips.shape[1] if observable_flips.ndim > 1 else 1
        pred_flat = predictions[:, :n_obs] if predictions.ndim > 1 else predictions.reshape(-1, 1)
        obs_flat = observable_flips if observable_flips.ndim > 1 else observable_flips.reshape(-1, 1)

        logical_errors = int(np.sum(np.any(pred_flat != obs_flat, axis=1)))
        ler = logical_errors / shots
        ci_low, ci_high = wilson_score_ci(logical_errors, shots)

        return BiasPointResult(
            bias_eta=eta,
            schedule_type=schedule_type.value,
            total_errors=logical_errors,
            total_shots=shots,
            ler=ler,
            ci_lower=ci_low,
            ci_upper=ci_high,
        )

    def _run_adaptive_schedule(
        self,
        eta: float,
        p_x: float,
        p_z: float,
    ) -> BiasPointResult:
        """Run experiment with adaptive stabilizer scheduling."""
        scheduler = AdaptiveXZScheduler(
            AdaptiveSchedulerConfig(
                ewma_alpha=self._config.ewma_alpha,
                theta_enter=self._config.theta_enter,
                theta_exit=self._config.theta_exit,
            )
        )

        shots_per_window = self._config.shots_per_schedule // self._config.num_windows
        total_errors = 0
        total_shots = 0

        import pymatching

        for window_idx in range(self._config.num_windows):
            current_sched = scheduler.current_schedule
            circuit = self._build_biased_circuit(p_x, p_z, schedule_type=current_sched.schedule_type)
            dem = circuit.detector_error_model(decompose_errors=True)
            matcher = pymatching.Matching.from_detector_error_model(dem)

            sampler = circuit.compile_detector_sampler()
            detection_events, observable_flips = sampler.sample(
                shots=shots_per_window,
                separate_observables=True,
            )

            # Extract detector coordinates to separate X and Z checks correctly
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

            obs = DefectObservation(
                round_idx=window_idx,
                x_defects=x_det,
                z_defects=z_det,
                total_x_stabilizers=len(x_indices),
                total_z_stabilizers=len(z_indices),
            )
            scheduler.update(obs)

            # Decode
            predictions = matcher.decode_batch(detection_events)
            n_obs = observable_flips.shape[1] if observable_flips.ndim > 1 else 1
            pred_flat = predictions[:, :n_obs] if predictions.ndim > 1 else predictions.reshape(-1, 1)
            obs_flat = observable_flips if observable_flips.ndim > 1 else observable_flips.reshape(-1, 1)

            logical_errors = int(np.sum(np.any(pred_flat != obs_flat, axis=1)))
            total_errors += logical_errors
            total_shots += shots_per_window

        ler = total_errors / total_shots if total_shots > 0 else 0.0
        ci_low, ci_high = wilson_score_ci(total_errors, total_shots)

        return BiasPointResult(
            bias_eta=eta,
            schedule_type="adaptive",
            total_errors=total_errors,
            total_shots=total_shots,
            ler=ler,
            ci_lower=ci_low,
            ci_upper=ci_high,
            adaptive_switches=scheduler.switch_count,
        )


    def _save_results(self) -> None:
        """Save experiment results to disk."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self._config.output_dir) / timestamp
