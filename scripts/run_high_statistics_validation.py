"""
High-Statistics Power Test: 50,000 shots per arm on distance-3 surface code.
Establishes statistical power and p-value for adaptive vs static QEC under non-stationary noise.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time

from adaptive_qec.experiments.adaptive_vs_static import (
    NoiseSchedule,
    run_adaptive_vs_static,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("high_statistics_validation")

def main():
    logger.info("Initializing 50,000-shot high-statistics validation experiment...")
