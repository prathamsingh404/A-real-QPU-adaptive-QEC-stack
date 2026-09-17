"""
CLI entry point for AdaptiveQEC.

Provides the `aqec` command-line interface registered in pyproject.toml.

Usage:
    aqec run --config configs/default.yaml
    aqec run --config configs/default.yaml --shots 50000
    aqec run --config configs/default.yaml --distances 3,5,7
    aqec serve --port 8000
"""

from __future__ import annotations

import logging
import sys

import click


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
@click.pass_context
def cli(ctx: click.Context, log_level: str) -> None:
    """AdaptiveQEC — Real-QPU Adaptive QEC Platform."""
    setup_logging(log_level)
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level


@cli.command()
@click.option("--config", default="configs/default.yaml", help="Path to configuration YAML file")
@click.option("--shots", default=None, type=int, help="Override shot count from config")
@click.option("--distances", default=None, help="Comma-separated distances for sweep (e.g., 3,5,7)")
@click.option("--temporal", default=None, type=int, help="Number of temporal experiments for drift analysis")
@click.option("--interval", default=60.0, type=float, help="Interval between temporal experiments (seconds)")
def run(config: str, shots: int | None, distances: str | None, temporal: int | None, interval: float) -> None:
    """Run a QEC experiment."""
    from adaptive_qec.config import load_config
    from adaptive_qec.experiment.manager import ExperimentManager

    logger = logging.getLogger("adaptive_qec.cli")
    logger.info("AdaptiveQEC — Real-QPU Adaptive QEC Platform")
    logger.info(f"Config: {config}")

    cfg = load_config(config)
    logger.info(
        f"Loaded config: code={cfg.qec.code.value}, "
        f"d={cfg.qec.distance}, R={cfg.qec.rounds}, "
        f"backend={cfg.hardware.backend}, channel={cfg.hardware.channel}"
    )

    manager = ExperimentManager(cfg)

    logger.info("Connecting to QPU...")
    manager.connect_qpu()

    if distances:
        distance_list = [int(d.strip()) for d in distances.split(",")]
        logger.info(f"Running distance sweep: {distance_list}")
        results = manager.run_distance_sweep(distance_list, shots=shots)

        logger.info("\n=== DISTANCE SWEEP RESULTS ===")
        for d, r in zip(distance_list, results):
            logger.info(
                f"  d={d}: LER={r.logical_error_rate:.6f} "
                f"[{r.logical_error_rate_ci_low:.6f}, {r.logical_error_rate_ci_high:.6f}]"
            )

    elif temporal:
        logger.info(f"Running temporal series: {temporal} experiments")
        results = manager.run_temporal_series(
            num_experiments=temporal,
            interval_seconds=interval,
            shots=shots,
        )

        logger.info("\n=== TEMPORAL SERIES RESULTS ===")
        for i, r in enumerate(results):
            logger.info(
                f"  Experiment {i+1}: LER={r.logical_error_rate:.6f}, "
                f"drift={r.drift_report.get('status', 'unknown')}"
            )

    else:
        metrics = manager.run_experiment(shots=shots)
        logger.info(
            f"\nResult: LER = {metrics.logical_error_rate:.6f} "
            f"[{metrics.logical_error_rate_ci_low:.6f}, "
            f"{metrics.logical_error_rate_ci_high:.6f}]"
        )


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
