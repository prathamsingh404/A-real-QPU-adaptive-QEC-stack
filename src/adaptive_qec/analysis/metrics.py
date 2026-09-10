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
    error_rate = num_errors / num_shots if num_shots > 0 else 0.0
    ci_low, ci_high = compute_confidence_interval(num_errors, num_shots, confidence)

    # Per-round error rate: 1 - (1 - p_L)^(1/R)
    per_round = None
    if num_rounds > 1 and error_rate < 1.0:
        per_round = 1 - (1 - error_rate) ** (1 / num_rounds)

    # Lambda ratio
    lambda_ratio = None
    if physical_error_rate and physical_error_rate > 0:
        lambda_ratio = error_rate / physical_error_rate

    return LogicalErrorMetrics(
        error_rate=error_rate,
        ci_low=ci_low,
        ci_high=ci_high,
        num_errors=num_errors,
        num_shots=num_shots,
        confidence=confidence,
        per_round_error_rate=per_round,
        lambda_ratio=lambda_ratio,
    )


@dataclass
class ThresholdEstimate:
    """Threshold estimation from distance scaling."""
    threshold: Optional[float] = None
    threshold_ci: Optional[tuple[float, float]] = None
    scaling_exponent: Optional[float] = None
    distances: list[int] = field(default_factory=list)
    error_rates: list[float] = field(default_factory=list)
    fit_quality: float = 0.0  # R²


def estimate_threshold(
    distances: list[int],
    error_rates: list[float],
    error_bars: Optional[list[tuple[float, float]]] = None,
) -> ThresholdEstimate:
    """
    Estimate the QEC threshold from logical error rates at multiple distances.

    Uses the scaling ansatz:
        p_L(d) = A * (p / p_th)^(d/2)

    where p_th is the threshold.

    Args:
        distances: list of code distances.
        error_rates: corresponding logical error rates.
        error_bars: optional (low, high) CI for each rate.

    Returns:
        ThresholdEstimate with fitted threshold.
    """
    if len(distances) < 2:
