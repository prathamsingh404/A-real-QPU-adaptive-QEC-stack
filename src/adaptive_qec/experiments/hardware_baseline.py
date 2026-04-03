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
