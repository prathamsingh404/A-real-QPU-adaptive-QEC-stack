#!/usr/bin/env python3
"""
Pre-Registered Statistical Power Analysis Script.
Task: TASK-082 [Point 93]
Ensures >= 80% statistical power at alpha = 0.05 for confirmatory trajectory-paired comparisons.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from adaptive_qec.analysis.statistical_rigor import (
    ConfirmatoryProtocol,
    PowerAnalysis,
    SeedPartition,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PowerAnalysis")


def run_power_analysis() -> dict:
    logger.info("Running Pre-Registered Confirmatory Statistical Power Analysis...")

    # 1. Verify seed partition integrity
    SeedPartition.validate_no_overlap()
    train_seeds = SeedPartition.get_seeds("train")
    val_seeds = SeedPartition.get_seeds("validation")
    test_seeds = SeedPartition.get_seeds("test")
