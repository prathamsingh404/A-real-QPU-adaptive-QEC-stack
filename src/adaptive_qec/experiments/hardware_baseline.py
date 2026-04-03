"""
Hardware baseline experiment.

Establishes the reference baseline by running QEC circuits on real
IBM hardware (or dry-run simulation). Captures:
    1. Baseline logical error rate with fixed MWPM decoder
    2. Hardware calibration snapshot (T1, T2, gate errors)
    3. Syndrome defect rate statistics (X vs Z balance)
    4. Decoder latency profiling (P50/P95/P99)

This baseline is essential for:
    - Validating simulator accuracy against real hardware
    - Establishing the "static floor" that adaptive must beat
    - Characterizing the noise bias of the target device
    - Sizing shot budgets for the full experiment

Usage:
    python -m adaptive_qec.experiments.hardware_baseline
    python -m adaptive_qec.experiments.hardware_baseline --dry-run
    python -m adaptive_qec.experiments.hardware_baseline --backend ibm_marrakesh
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

from adaptive_qec.runtime.budget import BudgetConfig, ShotBudgetManager
from adaptive_qec.runtime.qiskit_loop import (
    QiskitRuntimeLoop,
    RuntimeLoopConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class HardwareBaselineConfig:
    """Configuration for the hardware baseline experiment."""

    # Hardware
    backend_name: str = "ibm_marrakesh"
    dry_run: bool = True

    # QEC circuit
    code_distance: int = 3
    num_rounds: int = 4

    # Execution
    shots_per_batch: int = 1000
    num_batches: int = 20
    max_total_shots: int = 50000

    # Budget
    budget_max_shots: int = 50000
    budget_safety_margin: int = 2000

    # Output
    output_dir: str = "experiments/results/hardware_baseline"
    seed: int = 42


class HardwareBaselineExperiment:
    """
    Hardware baseline characterization experiment.

    Runs fixed-configuration QEC on real hardware to establish
    the static performance floor and characterize device noise.
    """

    def __init__(
        self,
        config: Optional[HardwareBaselineConfig] = None,
    ) -> None:
        self._config = config or HardwareBaselineConfig()

    def run(self) -> dict[str, Any]:
        """Execute the baseline experiment."""
        cfg = self._config
        t_start = time.time()

        logger.info(
            f"Starting hardware baseline: "
            f"backend={cfg.backend_name}, "
            f"d={cfg.code_distance}, "
            f"dry_run={cfg.dry_run}"
        )

        # Set up budget
        budget = ShotBudgetManager(BudgetConfig(
            max_total_shots=cfg.budget_max_shots,
            safety_margin=cfg.budget_safety_margin,
        ))

        # Set up runtime loop
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = str(Path(cfg.output_dir) / timestamp)

        loop_config = RuntimeLoopConfig(
            backend_name=cfg.backend_name,
            shots_per_batch=cfg.shots_per_batch,
            max_batches=cfg.num_batches,
            max_total_shots=cfg.max_total_shots,
            dry_run=cfg.dry_run,
            output_dir=output_dir,
        )

        loop = QiskitRuntimeLoop(
            config=loop_config,
            budget_manager=budget,
        )

        # Connect and run
        loop.connect()
        if not cfg.dry_run:
            loop.open_session()

        results = loop.run()

        elapsed = time.time() - t_start

        # Analyze results
        analysis = self._analyze(results)
        analysis["config"] = {
            "backend": cfg.backend_name,
            "code_distance": cfg.code_distance,
            "num_rounds": cfg.num_rounds,
            "dry_run": cfg.dry_run,
            "total_elapsed_s": round(elapsed, 2),
        }
        analysis["budget"] = budget.summary()

        # Save analysis
        output_path = Path(output_dir) / "baseline_analysis.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2)

        logger.info(f"Baseline analysis saved to {output_path}")
        self._print_summary(analysis)

        return analysis

    def _analyze(self, results: list) -> dict[str, Any]:
        """Analyze batch results for baseline characterization."""
        if not results:
            return {"error": "No results collected"}

        lers = [r.logical_error_rate for r in results]
        exec_times = [r.execution_time_s for r in results]
        total_shots = sum(r.shots for r in results)
        total_errors = sum(r.logical_errors for r in results)

        from adaptive_qec.analysis.significance import wilson_score_ci
        ci_low, ci_high = wilson_score_ci(total_errors, total_shots)

        return {
            "summary": {
                "total_batches": len(results),
                "total_shots": total_shots,
                "total_errors": total_errors,
                "overall_ler": total_errors / total_shots if total_shots > 0 else 0.0,
                "ci_95_lower": ci_low,
                "ci_95_upper": ci_high,
