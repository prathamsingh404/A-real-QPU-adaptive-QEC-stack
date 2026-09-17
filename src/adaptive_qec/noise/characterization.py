"""
Full noise characterization — combines calibration data with detector statistics.

Builds a comprehensive noise profile of the hardware state at experiment time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from adaptive_qec.noise.statistics import (
    DetectorStatistics,
    SpatialCorrelation,
    TemporalCorrelation,
    compute_detector_statistics,
    compute_spatial_correlation,
    compute_temporal_correlation,
)
from adaptive_qec.qpu.base import CalibrationSnapshot

logger = logging.getLogger(__name__)


@dataclass
class NoiseProfile:
    """
    Complete noise profile combining calibration and experimental data.

    This is the unified noise representation used by the adaptive decoder
    and the noise estimator.

    θ_t = {p_X, p_Y, p_Z, p_readout, p_leakage, ...}
    """
    timestamp: str
    backend: str

    # From calibration
    t1_values: Optional[np.ndarray] = None
    t2_values: Optional[np.ndarray] = None
    readout_errors: Optional[np.ndarray] = None
    gate_errors_1q: Optional[np.ndarray] = None
    gate_errors_2q: Optional[dict[tuple[int, int], float]] = None

    # From detector analysis
    detector_stats: Optional[DetectorStatistics] = None
    temporal_corr: Optional[TemporalCorrelation] = None
    spatial_corr: Optional[SpatialCorrelation] = None

    # Estimated noise parameters
    estimated_physical_error_rate: Optional[float] = None
    estimated_readout_error_rate: Optional[float] = None
    estimated_depolarizing_rate: Optional[float] = None
    correlated_noise_detected: bool = False
    leakage_detected: bool = False

    # Summary
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary."""
        d: dict[str, Any] = {
            "timestamp": self.timestamp,
            "backend": self.backend,
            "estimated_physical_error_rate": self.estimated_physical_error_rate,
            "estimated_readout_error_rate": self.estimated_readout_error_rate,
            "estimated_depolarizing_rate": self.estimated_depolarizing_rate,
            "correlated_noise_detected": self.correlated_noise_detected,
            "leakage_detected": self.leakage_detected,
        }

        if self.detector_stats:
            d["detector_stats"] = {
                "mean_detection_rate": self.detector_stats.mean_detection_rate,
                "max_detection_rate": self.detector_stats.max_detection_rate,
                "std_detection_rate": self.detector_stats.std_detection_rate,
                "num_hotspots": len(self.detector_stats.hotspot_detectors),
                "hotspot_detectors": self.detector_stats.hotspot_detectors[:20],
            }

        if self.temporal_corr:
            d["temporal_correlation"] = {
                "mean_lag1_correlation": (
                    float(self.temporal_corr.mean_autocorrelation[0])
                    if len(self.temporal_corr.mean_autocorrelation) > 0
                    else 0.0
                ),
                "num_persistent_detectors": len(self.temporal_corr.persistence_detectors),
            }

        if self.spatial_corr:
            d["spatial_correlation"] = {
                "num_significant_pairs": len(self.spatial_corr.significant_pairs),
            }

        d["summary"] = self.summary
        return d


class NoiseCharacterizer:
    """
    Builds a comprehensive noise profile from calibration and experimental data.

    Combines:
    1. Hardware calibration (T1, T2, readout, gate errors)
    2. Detector statistics (rates, correlations)
    3. Temporal correlation analysis
    4. Spatial correlation analysis
    5. Noise parameter estimation
    """

    def characterize(
        self,
        detection_events: np.ndarray,
        num_rounds: int,
        calibration: Optional[CalibrationSnapshot] = None,
        detector_coordinates: Optional[np.ndarray] = None,
    ) -> NoiseProfile:
        """
        Build a complete noise profile.

        Args:
            detection_events: shape (shots, num_detectors)
            num_rounds: number of QEC rounds
            calibration: optional calibration snapshot
            detector_coordinates: optional detector coordinates

        Returns:
            NoiseProfile with all analyses.
