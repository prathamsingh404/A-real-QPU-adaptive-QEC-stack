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


class NoiseSnapshot:
    """Instantaneous noise parameters at time step t.

    All values are from real calibration ranges.
    """
    def __init__(
        self,
        step: int = 0,
        p_1q: float = 0.000454,
        p_2q: float = 0.003021,
        p_ro: float = 0.01208,
        t1_mean_us: float = 180.0,
        t2_mean_us: float = 120.0,
        burst_active: bool = False,
        scenario_label: str = "",
        gate_error_1q: Optional[float] = None,
        gate_error_2q: Optional[float] = None,
        readout_error: Optional[float] = None,
        t1_us: Optional[float] = None,
        t2_us: Optional[float] = None,
    ) -> None:
        self.step = step
        self.p_1q = gate_error_1q if gate_error_1q is not None else p_1q
        self.p_2q = gate_error_2q if gate_error_2q is not None else p_2q
        self.p_ro = readout_error if readout_error is not None else p_ro
        self.t1_mean_us = t1_us if t1_us is not None else t1_mean_us
        self.t2_mean_us = t2_us if t2_us is not None else t2_mean_us
        self.burst_active = burst_active
        self.scenario_label = scenario_label

    @property
    def gate_error_1q(self) -> float:
        return self.p_1q

    @property
    def gate_error_2q(self) -> float:
        return self.p_2q

    @property
    def readout_error(self) -> float:
        return self.p_ro

    @property
    def t1_us(self) -> float:
        return self.t1_mean_us

    @property
    def t2_us(self) -> float:
        return self.t2_mean_us
