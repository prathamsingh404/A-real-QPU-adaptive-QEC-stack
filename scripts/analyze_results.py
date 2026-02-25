"""
Post-hoc analysis of saved experiments.

Usage:
    python scripts/analyze_results.py --experiment <experiment_id>
    python scripts/analyze_results.py --compare <id1>,<id2>
    python scripts/analyze_results.py --list
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_qec.data.store import ExperimentStore
from adaptive_qec.analysis.statistics import compare_error_rates


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
