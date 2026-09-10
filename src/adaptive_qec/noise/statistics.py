"""
Detector statistics and correlation analysis.

Calculates:
    - Single-detector rates: P(D_i = 1)
    - Pair correlations: C_ij = E[D_i D_j] - E[D_i]E[D_j]
    - Temporal autocorrelation: C(k) = corr(D_t, D_{t+k})
    - Spatial correlation maps

This is a major subsystem, not an afterthought.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DetectorStatistics:
    """Complete detector statistics for an experiment."""
    num_detectors: int
    num_shots: int
    detection_rates: np.ndarray          # P(D_i = 1), shape (num_detectors,)
    pair_correlations: np.ndarray        # C_ij, shape (num_detectors, num_detectors)
    mean_detection_rate: float
    max_detection_rate: float
    min_detection_rate: float
    std_detection_rate: float
    hotspot_detectors: list[int] = field(default_factory=list)  # unusually high rate


@dataclass
class TemporalCorrelation:
    """Temporal autocorrelation analysis."""
    max_lag: int
    autocorrelation: np.ndarray          # shape (num_detectors, max_lag)
    mean_autocorrelation: np.ndarray     # shape (max_lag,)
    persistence_detectors: list[int] = field(default_factory=list)


@dataclass
class SpatialCorrelation:
    """Spatial correlation analysis."""
    num_detectors: int
    correlation_matrix: np.ndarray       # shape (num_detectors, num_detectors)
    significant_pairs: list[tuple[int, int, float]] = field(default_factory=list)
    clustering_coefficient: float = 0.0


def compute_detector_statistics(
    detection_events: np.ndarray,
    significance_threshold: float = 2.5,
) -> DetectorStatistics:
    """
    Compute comprehensive detector statistics.

    Args:
        detection_events: shape (shots, num_detectors), binary.
        significance_threshold: number of std devs to flag a hotspot.

    Returns:
        DetectorStatistics with rates, correlations, and hotspots.
    """
    shots, num_detectors = detection_events.shape

    # P(D_i = 1)
    detection_rates = detection_events.mean(axis=0)

    # Pair correlations: C_ij = E[D_i D_j] - E[D_i]E[D_j]
    # Compute covariance matrix efficiently
    detection_float = detection_events.astype(np.float64)
    mean_vec = detection_rates
    centered = detection_float - mean_vec[np.newaxis, :]

    # Only compute upper triangle for large detector counts
    if num_detectors <= 1000:
        pair_correlations = (centered.T @ centered) / shots
    else:
        # Sparse computation for large codes
        logger.info(f"Large detector count ({num_detectors}), using sparse correlation")
        pair_correlations = np.zeros((num_detectors, num_detectors))
        batch = 100
        for i in range(0, num_detectors, batch):
            i_end = min(i + batch, num_detectors)
            for j in range(i, num_detectors, batch):
                j_end = min(j + batch, num_detectors)
                block = (centered[:, i:i_end].T @ centered[:, j:j_end]) / shots
                pair_correlations[i:i_end, j:j_end] = block
                if i != j:
                    pair_correlations[j:j_end, i:i_end] = block.T

    # Identify hotspot detectors (unusually high detection rate)
    mean_rate = detection_rates.mean()
    std_rate = detection_rates.std()
    hotspots = []
    if std_rate > 0:
        z_scores = (detection_rates - mean_rate) / std_rate
        hotspots = list(np.where(z_scores > significance_threshold)[0])

    result = DetectorStatistics(
        num_detectors=num_detectors,
        num_shots=shots,
        detection_rates=detection_rates,
        pair_correlations=pair_correlations,
        mean_detection_rate=float(mean_rate),
        max_detection_rate=float(detection_rates.max()),
        min_detection_rate=float(detection_rates.min()),
        std_detection_rate=float(std_rate),
        hotspot_detectors=hotspots,
    )

    logger.info(
        f"Detector statistics: mean_rate={mean_rate:.6f}, "
        f"std={std_rate:.6f}, hotspots={len(hotspots)}"
