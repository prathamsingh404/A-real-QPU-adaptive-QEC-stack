"""
Counterfactual Decoding and Paired Evaluation Engine.

Provides mathematically rigorous paired evaluation of decoders and adaptive policies
on identical syndrome bitstreams (TASK-020, TASK-021, Point 28, Point 29).

Key Properties:
    1. Feeds the exact same syndrome stream s_{1:N} to all classical decoders.
    2. Completely eliminates shot-by-shot syndrome variance between arms.
    3. Computes pairwise decoder disagreement C_t (essential for Gate 2 of CG-CCADS).
    4. Computes paired deltas Delta_i = e_{i, adaptive} - e_{i, static} and
       paired statistical significance tests (paired t-test and Wilcoxon signed-rank).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from scipy import stats

from adaptive_qec.decoders.base import Decoder, DecoderMetrics

logger = logging.getLogger(__name__)


def wilson_score_interval(errors: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score interval for binomial proportions."""
    if total == 0:
        return 0.0, 0.0
    p = errors / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denom
    spread = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return float(max(0.0, centre - spread)), float(min(1.0, centre + spread))


@dataclass
class PairedEvaluationResult:
    """Rigorous paired evaluation metrics for a single decoder or policy arm."""
    arm_name: str
    total_shots: int
    num_logical_errors: int
    logical_error_rate: float
    ler_wilson_ci: tuple[float, float]
    predictions: np.ndarray  # shape (shots, observables)
    error_mask: np.ndarray   # shape (shots,), True where prediction != actual

    # Paired metrics relative to reference baseline (typically Factory MWPM B0)
    delta_errors: Optional[np.ndarray] = None  # +1 if worse than B0, -1 if better, 0 if identical
    mean_delta_ler: float = 0.0                # LER(arm) - LER(B0)
    delta_std_err: float = 0.0
    p_value_paired_t: float = 1.0
    p_value_wilcoxon: float = 1.0
    decode_time_s: float = 0.0
    latency_mean_us: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_name": self.arm_name,
            "total_shots": self.total_shots,
            "num_logical_errors": self.num_logical_errors,
            "logical_error_rate": self.logical_error_rate,
            "ler_wilson_ci": list(self.ler_wilson_ci),
            "mean_delta_ler": self.mean_delta_ler,
            "delta_std_err": self.delta_std_err,
            "p_value_paired_t": self.p_value_paired_t,
            "p_value_wilcoxon": self.p_value_wilcoxon,
            "decode_time_s": self.decode_time_s,
            "latency_mean_us": self.latency_mean_us,
        }
