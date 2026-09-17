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

