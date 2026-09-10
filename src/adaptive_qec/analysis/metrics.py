"""
Metrics computation for QEC experiments.

Computes:
    - Logical error rate with confidence intervals
    - Physical-to-logical error rate ratio (Λ)
    - Threshold estimation via distance scaling
    - Per-round error rate
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit

from adaptive_qec.analysis.statistics import compute_confidence_interval

logger = logging.getLogger(__name__)


@dataclass
class LogicalErrorMetrics:
    """Comprehensive logical error metrics."""
    error_rate: float
    ci_low: float
    ci_high: float
    num_errors: int
    num_shots: int
    confidence: float
    per_round_error_rate: Optional[float] = None
    lambda_ratio: Optional[float] = None  # p_logical / p_physical


def compute_logical_error_metrics(
    num_errors: int,
    num_shots: int,
    num_rounds: int = 1,
    physical_error_rate: Optional[float] = None,
    confidence: float = 0.95,
) -> LogicalErrorMetrics:
    """
    Compute logical error rate with confidence interval.

    Args:
        num_errors: number of logical errors observed.
        num_shots: total shots.
        num_rounds: QEC rounds (for per-round rate).
        physical_error_rate: estimated physical error rate (for Λ ratio).
        confidence: confidence level for CI.

    Returns:
        LogicalErrorMetrics with all computed values.
    """
