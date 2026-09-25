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

    # Multi-phase timing
    phase_durations: list[int] = field(default_factory=lambda: [30, 30, 10, 30])

    def validate(self) -> None:
        """Validate that parameters are in realistic ranges."""
        assert 0 < self.baseline_p_1q < 0.01, "p_1q out of IBM Heron range"
        assert 0 < self.baseline_p_2q < 0.05, "p_2q out of IBM Heron range"
        assert 0 < self.baseline_p_ro < 0.10, "p_ro out of IBM Heron range"
        assert 50 < self.baseline_t1_us < 500, "T1 out of IBM Heron range"
        assert 30 < self.baseline_t2_us < 400, "T2 out of IBM Heron range"


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

    def get_noise(self, step: int) -> NoiseSnapshot:
        """Get the noise snapshot for the given step."""
        generator = self._generators.get(self._config.scenario_type)
        if generator is None:
            raise ValueError(f"Unknown scenario: {self._config.scenario_type}")
        return generator(step)

    def _stationary(self, step: int) -> NoiseSnapshot:
        """Constant noise at baseline calibration values."""
        return NoiseSnapshot(
            step=step,
            p_1q=self._config.baseline_p_1q,
            p_2q=self._config.baseline_p_2q,
            p_ro=self._config.baseline_p_ro,
            t1_mean_us=self._config.baseline_t1_us,
            t2_mean_us=self._config.baseline_t2_us,
            scenario_label="stationary",
        )

    def _linear_drift(self, step: int) -> NoiseSnapshot:
        """Monotonic degradation of noise parameters."""
        p_2q = self._config.baseline_p_2q + self._config.drift_rate_p2q * step
        t1 = max(50.0, self._config.baseline_t1_us + self._config.drift_rate_t1 * step)
        t2 = max(30.0, min(t1, self._config.baseline_t2_us + self._config.drift_rate_t1 * 0.7 * step))

        return NoiseSnapshot(
            step=step,
            p_1q=self._config.baseline_p_1q,
            p_2q=min(p_2q, 0.05),
            p_ro=self._config.baseline_p_ro,
            t1_mean_us=t1,
            t2_mean_us=t2,
            scenario_label="linear_drift",
        )

    def _sinusoidal_drift(self, step: int) -> NoiseSnapshot:
        """Periodic noise variation (simulates diurnal temperature cycle)."""
        phase = 2.0 * np.pi * step / self._config.period_steps
        p_2q = self._config.baseline_p_2q + self._config.amplitude_p2q * np.sin(phase)
        t1 = self._config.baseline_t1_us + 20.0 * np.cos(phase)

        return NoiseSnapshot(
            step=step,
            p_1q=self._config.baseline_p_1q,
            p_2q=float(np.clip(p_2q, 0.001, 0.05)),
            p_ro=self._config.baseline_p_ro,
            t1_mean_us=float(np.clip(t1, 80.0, 300.0)),
            t2_mean_us=float(np.clip(min(t1, self._config.baseline_t2_us), 50.0, 200.0)),
            scenario_label="sinusoidal_drift",
        )

    def _burst(self, step: int) -> NoiseSnapshot:
        """Sudden TLF-like burst event at a known step."""
        in_burst = (
            self._config.burst_start <= step
            < self._config.burst_start + self._config.burst_duration
        )

        if in_burst:
            p_2q = self._config.baseline_p_2q * self._config.burst_multiplier
            p_ro = self._config.baseline_p_ro * 1.5
        else:
            p_2q = self._config.baseline_p_2q
            p_ro = self._config.baseline_p_ro

        return NoiseSnapshot(
            step=step,
            p_1q=self._config.baseline_p_1q,
            p_2q=min(p_2q, 0.05),
            p_ro=min(p_ro, 0.10),
            t1_mean_us=self._config.baseline_t1_us,
            t2_mean_us=self._config.baseline_t2_us,
            burst_active=in_burst,
            scenario_label="burst",
        )

    def _multi_phase(self, step: int) -> NoiseSnapshot:
        """Concatenation of: stable → linear drift → burst → recovery."""
        durations = self._config.phase_durations
        boundaries = np.cumsum(durations)

        if step < boundaries[0]:
            # Phase 1: stationary
            return self._stationary(step)
        elif step < boundaries[1]:
            # Phase 2: linear drift
            drift_step = step - boundaries[0]
            return self._linear_drift(drift_step)
        elif step < boundaries[2]:
            # Phase 3: burst
            burst_config = ScenarioConfig(
                scenario_type=ScenarioType.BURST,
                total_steps=durations[2],
                baseline_p_1q=self._config.baseline_p_1q,
                baseline_p_2q=self._config.baseline_p_2q * 1.5,  # already degraded
                baseline_p_ro=self._config.baseline_p_ro,
                baseline_t1_us=self._config.baseline_t1_us * 0.8,
                baseline_t2_us=self._config.baseline_t2_us * 0.8,
                burst_start=0,
                burst_duration=durations[2],
                burst_multiplier=self._config.burst_multiplier,
            )
            factory = NoiseScenarioFactory(burst_config)
            snap = factory.get_noise(step - int(boundaries[1]))
            snap.scenario_label = "multi_phase:burst"
            return snap
        else:
            # Phase 4: recovery (return to stationary)
            snap = self._stationary(step)
            snap.scenario_label = "multi_phase:recovery"
            return snap


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def stationary_scenario(total_steps: int = 100) -> NoiseScenarioFactory:
    """Create a stationary noise scenario with IBM Heron defaults."""
    return NoiseScenarioFactory(ScenarioConfig(
        scenario_type=ScenarioType.STATIONARY,
        total_steps=total_steps,
    ))


def drift_scenario(total_steps: int = 100, rate: float = 0.0001) -> NoiseScenarioFactory:
    """Create a linear-drift noise scenario."""
    return NoiseScenarioFactory(ScenarioConfig(
        scenario_type=ScenarioType.LINEAR_DRIFT,
        total_steps=total_steps,
        drift_rate_p2q=rate,
    ))


def burst_scenario(
    total_steps: int = 100,
    burst_at: int = 50,
    duration: int = 10,
) -> NoiseScenarioFactory:
    """Create a burst-injection noise scenario."""
    return NoiseScenarioFactory(ScenarioConfig(
        scenario_type=ScenarioType.BURST,
        total_steps=total_steps,
        burst_start=burst_at,
        burst_duration=duration,
    ))


def multi_phase_scenario(
    phase_durations: Optional[list[int]] = None,
) -> NoiseScenarioFactory:
    """Create a multi-phase scenario (stable→drift→burst→recovery)."""
    durations = phase_durations or [30, 30, 10, 30]
    return NoiseScenarioFactory(ScenarioConfig(
        scenario_type=ScenarioType.MULTI_PHASE,
        total_steps=sum(durations),
        phase_durations=durations,
    ))
