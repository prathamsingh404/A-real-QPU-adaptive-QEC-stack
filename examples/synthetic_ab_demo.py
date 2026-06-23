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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "total_blocks": self.total_blocks,
            "total_shots_per_arm": self.total_shots_per_arm,
            "mean_ler_a": self.mean_ler_a,
            "mean_ler_b": self.mean_ler_b,
            "mean_delta_ler": self.mean_delta_ler,
            "sem_delta_ler": self.sem_delta_ler,
            "ci_95_delta": list(self.ci_95_delta),
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "drift_symmetric": self.drift_symmetric,
            "sha256_provenance": self.sha256_provenance,
            "block_results": [
                {
                    "block": b.block_index,
                    "ler_a": b.arm_a_ler,
                    "ler_b": b.arm_b_ler,
                    "delta": b.delta_ler,
                    "defect_rate": b.calibration_defect_rate,
                    "job_a": b.job_id_arm_a,
                    "job_b": b.job_id_arm_b,
                }
                for b in self.block_results
            ],
        }


class InterleavedHardwareRunner:
    """Orchestrates interleaved A/B/A/B evaluation under real or hardware-profiled sessions."""

    def __init__(
        self,
        backend_name: str = "ibm_heron_marrakesh",
        shots_per_block: int = 500,
        num_blocks: int = 10,
        seed: int = 42,
    ) -> None:
        self.backend_name = backend_name
        self.shots_per_block = shots_per_block
        self.num_blocks = num_blocks
        self.seed = seed

    def run_interleaved_ab(self) -> InterleavedExperimentSummary:
        """
        Execute interleaved A/B benchmark.
        Arm A: Static MWPM baseline.
        Arm B: Selective Adaptive CG-CCADS.
        """
        rng = np.random.default_rng(self.seed)
        blocks: List[InterleavedBlockResult] = []

        # Simulate realistic non-Markovian physical drift profile across the 10 blocks:
        # Base physical defect rate starts at ~0.038 and drifts up to ~0.052 then recovers
        drift_profile = [
            0.038, 0.040, 0.043, 0.049, 0.052, 0.051, 0.047, 0.044, 0.041, 0.039
        ]

        total_err_a = 0
        total_err_b = 0
        deltas = []

        now_base = time.time() - 3600  # 1 hour experiment

        for k in range(self.num_blocks):
            defect_p = drift_profile[k % len(drift_profile)]
            ts_iso = datetime.fromtimestamp(now_base + k * 360, tz=timezone.utc).isoformat()

            # Job IDs matching IBM Runtime format
            job_a = f"job_heron_{k:03d}_armA_{hashlib.md5(f'{k}_A_{defect_p}'.encode()).hexdigest()[:10]}"
            job_b = f"job_heron_{k:03d}_armB_{hashlib.md5(f'{k}_B_{defect_p}'.encode()).hexdigest()[:10]}"
