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

    def to_calibration_dict(self) -> dict[str, float]:
        return {
            "p_1q": self.p_1q,
            "p_2q": self.p_2q,
            "p_ro": self.p_ro,
            "t1_mean_us": self.t1_mean_us,
            "t2_mean_us": self.t2_mean_us,
            "gate_error_1q": self.p_1q,
            "gate_error_2q": self.p_2q,
            "readout_error": self.p_ro,
            "t1_us": self.t1_mean_us,
            "t2_us": self.t2_mean_us,
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

    # Drift parameters
    drift_rate_p2q: float = 0.0001   # per-step increase in p_2q
    drift_rate_t1: float = -0.5      # per-step decrease in T1 (μs)

    # Sinusoidal parameters
    amplitude_p2q: float = 0.002     # peak-to-peak variation
    period_steps: int = 200          # oscillation period

    # Burst parameters
    burst_start: int = 50            # step when burst begins
    burst_duration: int = 10         # how many steps the burst lasts
    burst_multiplier: float = 3.0    # noise multiplier during burst
    burst_probability: Optional[float] = None

    # Multi-phase timing
    phase_durations: list[int] = field(default_factory=lambda: [30, 30, 10, 30])

    def validate(self) -> None:
        """Validate that parameters are in realistic ranges."""
        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {self.total_steps}")
        if not (0 < self.baseline_p_1q < 0.05):
            raise ValueError("p_1q out of IBM Heron range")
        if not (0 < self.baseline_p_2q < 0.05):
            raise ValueError("p_2q out of IBM Heron range")
        if not (0 < self.baseline_p_ro < 0.10):
            raise ValueError("p_ro out of IBM Heron range")
        if not (50 <= self.baseline_t1_us <= 500):
            raise ValueError("T1 out of IBM Heron range")
        if not (30 <= self.baseline_t2_us <= 400):
            raise ValueError("T2 out of IBM Heron range")



class NoiseScenarioFactory:
    """Factory that generates time-varying noise profiles.

    Usage:
        factory = NoiseScenarioFactory(ScenarioConfig(
            scenario_type=ScenarioType.LINEAR_DRIFT,
            total_steps=100,
        ))
        for step in range(100):
            snapshot = factory.get_noise(step)
    """

    def __init__(self, config: ScenarioConfig) -> None:
        config.validate()
        self._config = config
        self._generators = {
            ScenarioType.STATIONARY: self._stationary,
            ScenarioType.LINEAR_DRIFT: self._linear_drift,
            ScenarioType.SINUSOIDAL_DRIFT: self._sinusoidal_drift,
            ScenarioType.BURST: self._burst,
            ScenarioType.MULTI_PHASE: self._multi_phase,
        }
        self._current_step: int = 0

    def step(self) -> NoiseSnapshot:
