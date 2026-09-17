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
    BURST_EVENT = "burst_event"


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
