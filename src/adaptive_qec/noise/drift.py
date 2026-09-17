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
