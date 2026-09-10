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
    )
    return result


def compute_temporal_correlation(
    detection_events: np.ndarray,
    num_rounds: int,
    max_lag: int = 10,
) -> TemporalCorrelation:
    """
    Compute temporal autocorrelation C(k) = corr(D_t, D_{t+k}).

    This reveals whether errors persist across QEC rounds, which is
    a signature of correlated noise, leakage, or drift.

    Args:
        detection_events: shape (shots, num_detectors)
        num_rounds: number of QEC rounds
        max_lag: maximum temporal lag to compute

    Returns:
        TemporalCorrelation with per-detector and mean autocorrelation.
    """
    shots, total_detectors = detection_events.shape
    detectors_per_round = total_detectors // (num_rounds + 1)

    if detectors_per_round == 0:
        logger.warning("Cannot compute temporal correlation: too few detectors per round")
        return TemporalCorrelation(
            max_lag=0,
            autocorrelation=np.array([]),
            mean_autocorrelation=np.array([]),
        )

    max_lag = min(max_lag, num_rounds)

    # Reshape to (shots, rounds+1, detectors_per_round)
    usable = detectors_per_round * (num_rounds + 1)
    reshaped = detection_events[:, :usable].reshape(
        shots, num_rounds + 1, detectors_per_round
    ).astype(np.float64)

    # Compute autocorrelation for each detector
    autocorr = np.zeros((detectors_per_round, max_lag))

    for d in range(detectors_per_round):
        series = reshaped[:, :, d]  # (shots, rounds+1)
        mean_d = series.mean()
        var_d = series.var()

        if var_d < 1e-12:
            continue

        for k in range(1, max_lag + 1):
            if k >= num_rounds + 1:
                break
            # C(k) = E[D_t * D_{t+k}] - E[D_t]^2, normalized
            products = series[:, :-k] * series[:, k:]
            autocorr[d, k - 1] = (products.mean() - mean_d ** 2) / var_d

    mean_autocorr = autocorr.mean(axis=0)

    # Identify detectors with persistent correlations
    persistence_threshold = 0.1
    persistence_detectors = []
    for d in range(detectors_per_round):
        if autocorr[d, 0] > persistence_threshold:
            persistence_detectors.append(d)

    logger.info(
        f"Temporal correlation: max_lag={max_lag}, "
        f"persistent_detectors={len(persistence_detectors)}"
    )
    return TemporalCorrelation(
        max_lag=max_lag,
        autocorrelation=autocorr,
        mean_autocorrelation=mean_autocorr,
        persistence_detectors=persistence_detectors,
    )


def compute_spatial_correlation(
    detection_events: np.ndarray,
    detector_coordinates: Optional[np.ndarray] = None,
    significance_level: float = 0.01,
) -> SpatialCorrelation:
    """
    Compute spatial correlations between detectors.

    Identifies which qubits/errors influence one another.

    Args:
        detection_events: shape (shots, num_detectors)
        detector_coordinates: optional (num_detectors, ndim) for distance weighting
        significance_level: p-value threshold for significant correlations

    Returns:
        SpatialCorrelation with correlation matrix and significant pairs.
    """
    shots, num_detectors = detection_events.shape
    detection_float = detection_events.astype(np.float64)

    # Pearson correlation matrix
    if num_detectors <= 500:
        correlation_matrix = np.corrcoef(detection_float.T)
        # Handle NaN from zero-variance detectors
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
    else:
        # Approximate for large detector counts
        correlation_matrix = np.zeros((num_detectors, num_detectors))
        means = detection_float.mean(axis=0)
        stds = detection_float.std(axis=0)
        stds[stds < 1e-12] = 1.0  # avoid division by zero

        batch = 100
        for i in range(0, num_detectors, batch):
            i_end = min(i + batch, num_detectors)
            normed_i = (detection_float[:, i:i_end] - means[i:i_end]) / stds[i:i_end]
            for j in range(i, num_detectors, batch):
                j_end = min(j + batch, num_detectors)
                normed_j = (detection_float[:, j:j_end] - means[j:j_end]) / stds[j:j_end]
                block = (normed_i.T @ normed_j) / shots
                correlation_matrix[i:i_end, j:j_end] = block
                if i != j:
                    correlation_matrix[j:j_end, i:i_end] = block.T

    # Find significant pairs (above threshold, excluding diagonal)
    significant_pairs = []
    # Use Bonferroni correction for multiple testing
    corrected_alpha = significance_level / max(1, num_detectors * (num_detectors - 1) // 2)
    # Critical correlation value for significance
    t_crit = stats.t.ppf(1 - corrected_alpha / 2, df=shots - 2)
    r_crit = t_crit / np.sqrt(t_crit ** 2 + shots - 2)

    for i in range(num_detectors):
        for j in range(i + 1, min(i + 50, num_detectors)):  # limit search range
            r = abs(correlation_matrix[i, j])
            if r > r_crit:
                significant_pairs.append((i, j, float(correlation_matrix[i, j])))

    logger.info(
        f"Spatial correlation: {num_detectors} detectors, "
        f"{len(significant_pairs)} significant pairs"
    )
    return SpatialCorrelation(
        num_detectors=num_detectors,
        correlation_matrix=correlation_matrix,
        significant_pairs=significant_pairs,
    )
