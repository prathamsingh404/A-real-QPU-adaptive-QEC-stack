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
            },
            "ler_statistics": {
                "mean": float(np.mean(lers)),
                "std": float(np.std(lers)),
                "median": float(np.median(lers)),
                "min": float(np.min(lers)),
                "max": float(np.max(lers)),
                "p25": float(np.percentile(lers, 25)),
                "p75": float(np.percentile(lers, 75)),
            },
            "timing": {
                "mean_batch_time_s": float(np.mean(exec_times)),
                "p50_batch_time_s": float(np.median(exec_times)),
                "p95_batch_time_s": float(np.percentile(exec_times, 95)),
                "p99_batch_time_s": float(np.percentile(exec_times, 99)),
                "total_time_s": float(np.sum(exec_times)),
            },
            "per_batch_ler": [round(l, 6) for l in lers],
        }

    def _print_summary(self, analysis: dict[str, Any]) -> None:
        """Print a human-readable summary."""
        s = analysis.get("summary", {})
        t = analysis.get("timing", {})

        print(f"\n{'='*50}")
        print(f"HARDWARE BASELINE RESULTS")
        print(f"{'='*50}")
        print(f"Backend:         {analysis['config']['backend']}")
        print(f"Dry run:         {analysis['config']['dry_run']}")
        print(f"Total shots:     {s.get('total_shots', 0):,}")
        print(f"Total errors:    {s.get('total_errors', 0):,}")
        print(f"Overall LER:     {s.get('overall_ler', 0):.6f}")
        print(f"95% CI:          [{s.get('ci_95_lower', 0):.6f}, "
              f"{s.get('ci_95_upper', 0):.6f}]")
        print(f"Avg batch time:  {t.get('mean_batch_time_s', 0):.3f}s")
        print(f"Total time:      {analysis['config']['total_elapsed_s']:.1f}s")
        print(f"{'='*50}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hardware baseline experiment"
    )
    parser.add_argument("--backend", type=str, default="ibm_marrakesh")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--output", type=str,
                        default="experiments/results/hardware_baseline")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = HardwareBaselineConfig(
        backend_name=args.backend,
        code_distance=args.distance,
        num_batches=args.batches,
        shots_per_batch=args.shots,
        dry_run=args.dry_run,
        output_dir=args.output,
    )

    experiment = HardwareBaselineExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
