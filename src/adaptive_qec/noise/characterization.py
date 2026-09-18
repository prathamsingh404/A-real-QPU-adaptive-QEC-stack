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
        """
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).isoformat()
        backend = calibration.backend_name if calibration else "unknown"

        logger.info("Starting noise characterization...")

        # 1. Detector statistics
        detector_stats = compute_detector_statistics(detection_events)

        # 2. Temporal correlation
        temporal_corr = compute_temporal_correlation(
            detection_events,
            num_rounds=num_rounds,
            max_lag=min(10, num_rounds),
        )

        # 3. Spatial correlation
        spatial_corr = compute_spatial_correlation(
            detection_events,
            detector_coordinates=detector_coordinates,
        )

        # 4. Build profile
        profile = NoiseProfile(
            timestamp=timestamp,
            backend=backend,
            detector_stats=detector_stats,
            temporal_corr=temporal_corr,
            spatial_corr=spatial_corr,
        )

        # 5. Add calibration data
        if calibration:
            profile.t1_values = calibration.t1_values()
            profile.t2_values = calibration.t2_values()
            profile.readout_errors = calibration.readout_errors()

            # Average readout error
            re = profile.readout_errors
            profile.estimated_readout_error_rate = float(re[re > 0].mean()) if np.any(re > 0) else 0.0

            # Extract 2Q gate errors
            profile.gate_errors_2q = {}
            for gc in calibration.gate_calibrations:
                if len(gc.qubits) == 2 and gc.error is not None:
                    profile.gate_errors_2q[gc.qubits] = gc.error

            # Estimate physical error rate from gate errors
            if profile.gate_errors_2q:
                errors_2q = list(profile.gate_errors_2q.values())
                profile.estimated_physical_error_rate = float(np.mean(errors_2q))
                profile.estimated_depolarizing_rate = profile.estimated_physical_error_rate

        # 6. Detect correlated noise signatures
        if temporal_corr and len(temporal_corr.persistence_detectors) > 0:
            persistence_fraction = (
                len(temporal_corr.persistence_detectors) /
                max(1, temporal_corr.autocorrelation.shape[0])
            )
            if persistence_fraction > 0.1:
                profile.correlated_noise_detected = True
                logger.warning(
                    f"Correlated noise detected: {len(temporal_corr.persistence_detectors)} "
                    f"persistent detectors ({persistence_fraction:.1%})"
                )

        # 7. Detect leakage signatures
        if num_rounds >= 3 and detection_events.ndim == 2:
            try:
                from adaptive_qec.noise.burst_detector import reshape_syndromes_to_tensor
                from adaptive_qec.noise.leakage import LeakageDetector
                det_per_round = detection_events.shape[1] // num_rounds
                if det_per_round > 0:
                    # Average over first few shots to build a stable syndrome trace
                    mean_trace = (detection_events[:min(10, len(detection_events))].mean(axis=0) > 0.3).astype(np.uint8)
                    syndrome_tensor = reshape_syndromes_to_tensor(
                        mean_trace, num_rounds, det_per_round
                    )
                    leakage_detector = LeakageDetector()
                    leakage_analysis = leakage_detector.analyze(syndrome_tensor)
                    if len(leakage_analysis.leaked_qubits) > 0:
                        profile.leakage_detected = True
            except Exception as e:
                logger.debug(f"Leakage check skipped: {e}")

        # 8. Summary
        profile.summary = {
            "mean_detection_rate": detector_stats.mean_detection_rate,
            "num_hotspots": len(detector_stats.hotspot_detectors),
            "correlated_noise": profile.correlated_noise_detected,
            "leakage": profile.leakage_detected,
            "estimated_physical_error": profile.estimated_physical_error_rate,
        }

        logger.info(f"Noise characterization complete: {profile.summary}")
        return profile
