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

