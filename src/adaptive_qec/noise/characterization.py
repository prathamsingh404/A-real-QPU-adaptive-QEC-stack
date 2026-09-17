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
