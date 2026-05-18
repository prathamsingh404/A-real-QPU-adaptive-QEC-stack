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
    # Predefined 12-scenario benchmark suite (TASK-063 to TASK-075)
    DO_NOTHING = "do_nothing"
    OBVIOUS_ADAPTATION = "obvious_adaptation"
    TOO_FAST_TO_ADAPT = "too_fast_to_adapt"
    SWITCHING_TRAP = "switching_trap"
    HIDDEN_TRANSITION = "hidden_transition"
    RECOVERY = "recovery"
    PERSISTENT_LEAKAGE = "persistent_leakage"
    DECEPTIVE_NOISE = "deceptive_noise"


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
            ScenarioType.DO_NOTHING: self._do_nothing,
            ScenarioType.OBVIOUS_ADAPTATION: self._obvious_adaptation,
            ScenarioType.TOO_FAST_TO_ADAPT: self._too_fast_to_adapt,
            ScenarioType.SWITCHING_TRAP: self._switching_trap,
            ScenarioType.HIDDEN_TRANSITION: self._hidden_transition,
            ScenarioType.RECOVERY: self._recovery,
            ScenarioType.PERSISTENT_LEAKAGE: self._persistent_leakage,
            ScenarioType.DECEPTIVE_NOISE: self._deceptive_noise,
        }
        self._current_step: int = 0

    def step(self) -> NoiseSnapshot:
        """Advance and return the next sequential noise snapshot."""
        snapshot = self.get_noise(self._current_step)
        self._current_step += 1
        return snapshot

    def reset(self) -> None:
        """Reset the sequential step counter."""
        self._current_step = 0

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
        """Sudden TLF-like burst event at a known step or probabilistic."""
        if self._config.burst_probability is not None:
            rng = np.random.default_rng(seed=step + 12345)
            in_burst = bool(rng.random() < self._config.burst_probability)
        else:
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

    def _do_nothing(self, step: int) -> NoiseSnapshot:
        """TASK-069: Flat stationary noise below threshold where optimal action is STAY."""
        return NoiseSnapshot(
            step=step,
            p_1q=0.0003,
            p_2q=0.0025,
            p_ro=0.008,
            t1_mean_us=220.0,
            t2_mean_us=160.0,
            scenario_label="do_nothing",
        )

    def _obvious_adaptation(self, step: int) -> NoiseSnapshot:
        """TASK-070: Step change at step 20 with strong persistent leakage."""
        if step < 20:
            return self._stationary(step)
        return NoiseSnapshot(
            step=step,
            p_1q=0.0006,
            p_2q=0.012,
            p_ro=0.02,
            t1_mean_us=110.0,
            t2_mean_us=60.0,
            scenario_label="obvious_adaptation",
        )

    def _too_fast_to_adapt(self, step: int) -> NoiseSnapshot:
        """TASK-071: High-frequency jitter changing every step."""
        rng = np.random.default_rng(seed=step * 10007)
        jitter = float(rng.uniform(-0.002, 0.006))
        return NoiseSnapshot(
            step=step,
            p_1q=0.0004,
            p_2q=float(np.clip(0.004 + jitter, 0.001, 0.03)),
            p_ro=0.012,
            t1_mean_us=180.0,
            t2_mean_us=120.0,
            scenario_label="too_fast_to_adapt",
        )

    def _switching_trap(self, step: int) -> NoiseSnapshot:
        """TASK-072: Alternating bistable drift swapping every 3 steps."""
        is_high = ((step // 3) % 2) == 1
        p_2q = 0.009 if is_high else 0.0025
        return NoiseSnapshot(
            step=step,
            p_1q=0.0004,
            p_2q=p_2q,
            p_ro=0.01,
            t1_mean_us=140.0 if is_high else 200.0,
            t2_mean_us=80.0 if is_high else 150.0,
            scenario_label="switching_trap",
        )

    def _hidden_transition(self, step: int) -> NoiseSnapshot:
        """TASK-073: Extremely slow subtle drift (delta p = 1e-5 * step)."""
        p_2q = 0.003 + 0.000015 * step
        return NoiseSnapshot(
            step=step,
            p_1q=0.00045,
            p_2q=float(min(p_2q, 0.015)),
            p_ro=0.012,
            t1_mean_us=max(80.0, 180.0 - 0.2 * step),
            t2_mean_us=max(50.0, 120.0 - 0.15 * step),
            scenario_label="hidden_transition",
        )

    def _recovery(self, step: int) -> NoiseSnapshot:
        """TASK-074: Burst at steps 10-20, returning to baseline from step 21."""
        in_burst = 10 <= step < 20
        return NoiseSnapshot(
            step=step,
            p_1q=0.0004,
            p_2q=0.014 if in_burst else 0.003,
            p_ro=0.03 if in_burst else 0.01,
            t1_mean_us=90.0 if in_burst else 180.0,
            t2_mean_us=50.0 if in_burst else 120.0,
            burst_active=in_burst,
            scenario_label="recovery:burst" if in_burst else "recovery:baseline",
        )

    def _persistent_leakage(self, step: int) -> NoiseSnapshot:
        """TASK-075: Correlated leakage persisting from step 15 onward."""
        leaked = step >= 15
        return NoiseSnapshot(
            step=step,
            p_1q=0.0004,
            p_2q=0.008 if leaked else 0.003,
            p_ro=0.015 if leaked else 0.01,
            t1_mean_us=120.0 if leaked else 180.0,
            t2_mean_us=70.0 if leaked else 120.0,
            scenario_label="persistent_leakage",
        )

    def _deceptive_noise(self, step: int) -> NoiseSnapshot:
        """TASK-068: Readout error shifts from 0.01 to 0.05 while gate error stays fixed."""
        high_ro = step >= 25
        return NoiseSnapshot(
            step=step,
            p_1q=0.0004,
            p_2q=0.003,  # Fixed gate error
            p_ro=0.055 if high_ro else 0.01,
            t1_mean_us=180.0,
            t2_mean_us=120.0,
            scenario_label="deceptive_noise",
        )


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
    burst_probability: Optional[float] = None,
    burst_magnitude: Optional[float] = None,
) -> NoiseScenarioFactory:
    """Create a burst-injection noise scenario."""
    multiplier = burst_magnitude if burst_magnitude is not None else 3.0
    return NoiseScenarioFactory(ScenarioConfig(
        scenario_type=ScenarioType.BURST,
        total_steps=total_steps,
        burst_start=burst_at,
        burst_duration=duration,
        burst_multiplier=multiplier,
        burst_probability=burst_probability,
    ))


def multi_phase_scenario(
    phase_durations: Optional[list[int]] = None,
    total_steps: Optional[int] = None,
) -> NoiseScenarioFactory:
    """Create a multi-phase scenario (stable→drift→burst→recovery)."""
    if total_steps is not None:
        q = total_steps // 4
        durations = [q, q, q, total_steps - 3 * q]
    else:
        durations = phase_durations or [30, 30, 10, 30]
    return NoiseScenarioFactory(ScenarioConfig(
        scenario_type=ScenarioType.MULTI_PHASE,
        total_steps=sum(durations),
        phase_durations=durations,
    ))


# ---------------------------------------------------------------------------
# Synthetic Environment & Observability Abstraction (TASK-065, TASK-066)
# ---------------------------------------------------------------------------

class ObservabilityLevel(str, Enum):
    """Four observability tiers demarcating controller visibility."""
    O0_RAW_BITSTRINGS = "O0"
    O1_CALIBRATION_TELEMETRY = "O1"
    O2_DEPLOYABLE_OBSERVABLES = "O2"
    O_ORACLE_FULL_LATENT = "O_oracle"


class SyntheticEnvironment:
    """
    Environment(latent_trajectory, observable_stream) abstraction (TASK-065, TASK-066).
    Enforces strict information boundaries according to the selected ObservabilityLevel.
    """

    def __init__(
        self,
        factory: NoiseScenarioFactory,
        observability: ObservabilityLevel = ObservabilityLevel.O2_DEPLOYABLE_OBSERVABLES,
        code_distance: int = 3,
    ) -> None:
        self.factory = factory
        self.observability = observability
        self.code_distance = code_distance
        self.step_idx: int = 0

    def step(self) -> tuple[dict[str, Any], NoiseSnapshot]:
        """Advance one environment step and return (observable_state, latent_snapshot)."""
        snap = self.factory.get_noise(self.step_idx)
        self.step_idx += 1

        # O0: Defect density and topological size only
        defect_rate = float(np.clip(snap.p_2q * 2.5 + snap.p_ro * 0.5, 0.001, 0.49))
        obs: dict[str, Any] = {
            "step": snap.step,
            "code_distance": self.code_distance,
            "defect_rate": defect_rate,
        }

        # O1: Add macro calibration snapshot
        if self.observability in (
            ObservabilityLevel.O1_CALIBRATION_TELEMETRY,
            ObservabilityLevel.O2_DEPLOYABLE_OBSERVABLES,
            ObservabilityLevel.O_ORACLE_FULL_LATENT,
        ):
            obs["t1_mean_us"] = snap.t1_mean_us
            obs["t2_mean_us"] = snap.t2_mean_us
            obs["p_2q"] = snap.p_2q
            obs["p_ro"] = snap.p_ro

        # O2: Add deployable signals (spatial defect rates, drift z-score, leakage signature)
        if self.observability in (
            ObservabilityLevel.O2_DEPLOYABLE_OBSERVABLES,
            ObservabilityLevel.O_ORACLE_FULL_LATENT,
        ):
            obs["drift_magnitude"] = float(abs(snap.p_2q - 0.003) * 1000.0)
            obs["leakage_fraction"] = 0.06 if "persistent_leakage" in snap.scenario_label else 0.0

        # O_oracle: Privileged latent variables (forbidden online, B6 upper bound only)
        if self.observability == ObservabilityLevel.O_ORACLE_FULL_LATENT:
            obs["latent_true_p_2q"] = snap.p_2q
            obs["latent_burst_active"] = snap.burst_active
            obs["latent_scenario"] = snap.scenario_label

        return obs, snap


