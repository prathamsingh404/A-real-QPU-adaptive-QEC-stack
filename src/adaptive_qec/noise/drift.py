"""
Drift detection system.

Explicitly identifies:
    stable → drift detected → magnitude → affected qubits → affected parameters

Methods:
    Simple:  EWMA, CUSUM
    ML:      isolation forest, change-point detection (V1+)

A simple statistical detector beating a complicated model would itself be useful.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DriftStatus(str, Enum):
    STABLE = "stable"
    WARNING = "warning"
    DRIFT_DETECTED = "drift_detected"
    SEVERE = "severe"


@dataclass
class DriftReport:
    """Complete drift analysis report."""
    status: DriftStatus
    magnitude: float                              # overall drift magnitude
    affected_detectors: list[int] = field(default_factory=list)
    affected_parameters: list[str] = field(default_factory=list)
    detector_drift_values: Optional[np.ndarray] = None  # per-detector drift metric
    change_points: list[int] = field(default_factory=list)  # indices where drift occurs
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        return {
            "status": self.status.value,
            "magnitude": float(self.magnitude),
            "affected_detectors": self.affected_detectors,
            "affected_parameters": self.affected_parameters,
            "change_points": self.change_points,
            "details": self.details,
        }


class EWMADriftDetector:
    """
    Exponentially Weighted Moving Average drift detector.

    Tracks p_i(t) over time using EWMA and detects when the
    smoothed rate deviates significantly from the baseline.

    Simple, interpretable, and effective for gradual drift.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        warning_sigma: float = 2.0,
        alarm_sigma: float = 3.0,
        severe_sigma: float = 5.0,
        warmup_samples: int = 20,
    ) -> None:
        """
        Args:
            alpha: EWMA smoothing factor (0 < alpha <= 1). Smaller = more smoothing.
            warning_sigma: standard deviations for warning threshold.
            alarm_sigma: standard deviations for drift alarm.
            severe_sigma: standard deviations for severe drift.
            warmup_samples: minimum samples before drift detection activates.
        """
        self._alpha = alpha
        self._warning_sigma = warning_sigma
        self._alarm_sigma = alarm_sigma
        self._severe_sigma = severe_sigma
        self._warmup = warmup_samples

        # State
        self._ewma: Optional[np.ndarray] = None
        self._baseline_mean: Optional[np.ndarray] = None
        self._baseline_var: Optional[np.ndarray] = None
        self._sample_count = 0
        self._history: list[np.ndarray] = []

    def update(self, detection_rates: np.ndarray) -> DriftReport:
        """
        Update the drift detector with new detection rates.

        Args:
            detection_rates: P(D_i = 1) for each detector from latest experiment.

        Returns:
            DriftReport with current status.
        """
        self._sample_count += 1
        self._history.append(detection_rates.copy())

        if self._ewma is None or self._ewma.shape != detection_rates.shape:
            self._ewma = detection_rates.copy()
            self._baseline_mean = detection_rates.copy()
            self._baseline_var = np.zeros_like(detection_rates)
            self._sample_count = 1
            self._history = [detection_rates.copy()]
            return DriftReport(
                status=DriftStatus.STABLE,
                magnitude=0.0,
                details={"message": "Initializing baseline"},
            )

        # Update EWMA
        self._ewma = self._alpha * detection_rates + (1 - self._alpha) * self._ewma

        # Update baseline statistics (Welford's online algorithm)
        if self._sample_count <= self._warmup:
            delta = detection_rates - self._baseline_mean
            self._baseline_mean += delta / self._sample_count
            delta2 = detection_rates - self._baseline_mean
            self._baseline_var += delta * delta2

            return DriftReport(
                status=DriftStatus.STABLE,
                magnitude=0.0,
                details={"message": f"Warmup: {self._sample_count}/{self._warmup}"},
            )

        # Compute baseline standard deviation
        baseline_std = np.sqrt(self._baseline_var / (self._warmup - 1))
        baseline_std[baseline_std < 1e-8] = 1e-8  # prevent division by zero

        # Compute z-scores for EWMA deviation from baseline
        # EWMA variance is reduced by factor alpha / (2 - alpha)
        ewma_std = baseline_std * np.sqrt(self._alpha / (2 - self._alpha))
        z_scores = np.abs(self._ewma - self._baseline_mean) / ewma_std

        # Classify drift
        max_z = float(z_scores.max())
        mean_z = float(z_scores.mean())

        severe_dets = list(np.where(z_scores > self._severe_sigma)[0])
        alarm_dets = list(np.where(z_scores > self._alarm_sigma)[0])
        warning_dets = list(np.where(z_scores > self._warning_sigma)[0])

        if len(severe_dets) > 0:
            status = DriftStatus.SEVERE
        elif len(alarm_dets) > 0:
            status = DriftStatus.DRIFT_DETECTED
        elif len(warning_dets) > 0:
            status = DriftStatus.WARNING
        else:
            status = DriftStatus.STABLE

        # Determine affected parameters
        affected_params = []
        if len(alarm_dets) > 0:
            affected_params.append("detection_rate")
            # Check if drift is in readout (all detectors) or specific region
            fraction_affected = len(alarm_dets) / len(detection_rates)
            if fraction_affected > 0.5:
                affected_params.append("global_noise")
            else:
                affected_params.append("local_noise")

        report = DriftReport(
            status=status,
            magnitude=max_z,
            affected_detectors=alarm_dets,
            affected_parameters=affected_params,
            detector_drift_values=z_scores,
            details={
                "max_z_score": max_z,
                "mean_z_score": mean_z,
                "num_warning": len(warning_dets),
                "num_alarm": len(alarm_dets),
                "num_severe": len(severe_dets),
                "sample_count": self._sample_count,
            },
        )

        if status != DriftStatus.STABLE:
            logger.warning(
                f"Drift {status.value}: magnitude={max_z:.2f}, "
                f"affected_detectors={len(alarm_dets)}"
            )

        return report


class CUSUMDriftDetector:
    """
    Cumulative Sum (CUSUM) drift detector.

    More sensitive to sudden shifts than EWMA. Tracks cumulative
    deviations from the baseline mean.
    """

    def __init__(
        self,
        threshold: float = 5.0,
        drift_allowance: float = 0.5,
        warmup_samples: int = 20,
    ) -> None:
        """
        Args:
            threshold: CUSUM decision threshold (h).
            drift_allowance: minimum shift to detect (k), in units of std.
            warmup_samples: samples for baseline estimation.
        """
        self._threshold = threshold
        self._allowance = drift_allowance
        self._warmup = warmup_samples

        # State
        self._s_plus: Optional[np.ndarray] = None   # upper CUSUM
        self._s_minus: Optional[np.ndarray] = None   # lower CUSUM
        self._baseline_mean: Optional[np.ndarray] = None
        self._baseline_std: Optional[np.ndarray] = None
        self._sample_count = 0
        self._history: list[np.ndarray] = []

    def update(self, detection_rates: np.ndarray) -> DriftReport:
        """
        Update CUSUM detector with new detection rates.

        Args:
            detection_rates: P(D_i = 1) for each detector.

        Returns:
            DriftReport.
        """
        self._sample_count += 1
        self._history.append(detection_rates.copy())

        num_detectors = len(detection_rates)

        if self._baseline_mean is None or self._baseline_mean.shape != detection_rates.shape:
            self._baseline_mean = detection_rates.copy()
            self._sample_count = 1
            self._history = [detection_rates.copy()]
            self._s_plus = None
            self._s_minus = None
            self._baseline_std = None
            return DriftReport(
                status=DriftStatus.STABLE,
                magnitude=0.0,
                details={"message": f"CUSUM warmup: 1/{self._warmup}"},
            )

        if self._sample_count <= self._warmup:
            delta = detection_rates - self._baseline_mean
            self._baseline_mean += delta / self._sample_count

            if self._sample_count == self._warmup:
                # Compute baseline standard deviation
                history_arr = np.array(self._history)
                self._baseline_std = history_arr.std(axis=0)
                self._baseline_std[self._baseline_std < 1e-8] = 1e-8
                self._s_plus = np.zeros(num_detectors)
                self._s_minus = np.zeros(num_detectors)

            return DriftReport(
                status=DriftStatus.STABLE,
                magnitude=0.0,
