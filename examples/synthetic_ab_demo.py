"""
Synthetic Interleaved A/B Benchmark Demonstration (NOT REAL HARDWARE).

PROVENANCE: SYNTHETIC SIMULATION ONLY.
This file is a numerical demo illustrating the mathematical concept of
interleaved A/B drift cancellation. It uses synthetic binomial draws
and a hardcoded noise profile to demonstrate how temporal interleaving
averages out background drift in statistical tests.

DO NOT CITE AS PHYSICAL HARDWARE EVIDENCE.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InterleavedBlockResult:
    """One matched temporal slice comparing Arm A (Static) vs Arm B (Adaptive)."""
    block_index: int
    timestamp_iso: str
    shots_per_arm: int
    arm_a_name: str
    arm_a_errors: int
    arm_a_ler: float
    arm_b_name: str
    arm_b_errors: int
    arm_b_ler: float
    delta_ler: float            # arm_b_ler - arm_a_ler (negative means Arm B superior)
    calibration_defect_rate: float
    job_id_arm_a: str
    job_id_arm_b: str


@dataclass
class InterleavedExperimentSummary:
    backend_name: str
    total_blocks: int
    total_shots_per_arm: int
    mean_ler_a: float
    mean_ler_b: float
    mean_delta_ler: float
    sem_delta_ler: float
    ci_95_delta: Tuple[float, float]
    t_statistic: float
    p_value: float
    drift_symmetric: bool       # Verifies that background drift impacted both arms equally
    block_results: List[InterleavedBlockResult] = field(default_factory=list)
    sha256_provenance: str = ""
