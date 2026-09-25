"""
Noise scenario factory for controlled ablation experiments.

Each scenario defines a temporal noise profile that can be applied
during an experiment to test controller behaviour under different
hardware conditions.  Scenarios use *real* calibration ranges from
IBM Heron processors, not arbitrary constants.

Scenarios:
    - Stationary: constant noise at calibrated values.
    - Linear drift: monotonic degradation of T1/T2.
    - Sinusoidal drift: periodic variation (diurnal cycle simulation).
    - Burst injection: sudden TLF-like noise events.
    - Multi-phase: concatenation of stable → drift → burst → recovery.

All parameters are drawn from measured IBM Heron ranges:
    T1 ∈ [80, 300] μs
    T2 ∈ [50, 200] μs
    CZ error ∈ [0.001, 0.015]
    Readout error ∈ [0.005, 0.03]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ScenarioType(str, Enum):
    STATIONARY = "stationary"
    LINEAR_DRIFT = "linear_drift"
    SINUSOIDAL_DRIFT = "sinusoidal_drift"
    BURST = "burst"
    MULTI_PHASE = "multi_phase"


@dataclass
class NoiseSnapshot:
    """Instantaneous noise parameters at time step t.

    All values are from real calibration ranges.
    """
    step: int
    p_1q: float        # single-qubit depolarizing error
    p_2q: float        # two-qubit depolarizing error
    p_ro: float        # readout error
    t1_mean_us: float  # mean T1 in microseconds
    t2_mean_us: float  # mean T2 in microseconds
    burst_active: bool = False
    scenario_label: str = ""

    def to_calibration_dict(self) -> dict[str, float]:
        return {
            "p_1q": self.p_1q,
            "p_2q": self.p_2q,
            "p_ro": self.p_ro,
            "t1_mean_us": self.t1_mean_us,
            "t2_mean_us": self.t2_mean_us,
        }


@dataclass
class ScenarioConfig:
    """Configuration for a noise scenario.

    Parameters are drawn from real IBM Heron calibration ranges.
    """
    scenario_type: ScenarioType
    total_steps: int

    # Baseline calibration (from real hardware)
    baseline_p_1q: float = 0.000454
    baseline_p_2q: float = 0.003021
    baseline_p_ro: float = 0.01208
    baseline_t1_us: float = 180.0
    baseline_t2_us: float = 120.0

