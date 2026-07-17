"""
CLI entry point for running QEC experiments.

Usage:
    python scripts/run_experiment.py --config configs/default.yaml
    python scripts/run_experiment.py --config configs/default.yaml --shots 50000
    python scripts/run_experiment.py --config configs/default.yaml --distances 3,5,7
    python scripts/run_experiment.py --config configs/default.yaml --temporal 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_qec.config import load_config
from adaptive_qec.experiment.manager import ExperimentManager


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AdaptiveQEC — Run QEC experiments on real QPU hardware"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="Override shot count from config",
    )
    parser.add_argument(
        "--distances",
        type=str,
        default=None,
        help="Comma-separated distances for sweep (e.g., 3,5,7)",
    )
    parser.add_argument(
        "--temporal",
        type=int,
        default=None,
        help="Number of temporal experiments for drift analysis",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Interval between temporal experiments (seconds)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger("adaptive_qec.cli")
    logger.info("AdaptiveQEC — Real-QPU Adaptive QEC Platform")
    logger.info(f"Config: {args.config}")

    # Load configuration
    config = load_config(args.config)
    logger.info(
        f"Loaded config: code={config.qec.code.value}, "
        f"d={config.qec.distance}, R={config.qec.rounds}, "
        f"backend={config.hardware.backend}"
    )

    # Create experiment manager
    manager = ExperimentManager(config)
