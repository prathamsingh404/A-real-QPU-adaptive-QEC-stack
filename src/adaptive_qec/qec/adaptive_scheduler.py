"""
Adaptive X/Z stabilizer scheduler.

Dynamically adjusts the temporal frequency of X-type vs Z-type stabilizer
measurements based on observed syndrome defect imbalance. This is the
implementation of the paper's second core novelty (Claim 3 in the roadmap):

    "Under noise with T1/T2 asymmetry, measuring the more-informative
     stabilizer type more frequently improves logical error rate."

Algorithm:
    1. Maintain an EWMA (Exponentially Weighted Moving Average) of
       X-type and Z-type defect rates from recent syndrome rounds.
    2. Compute the syndrome imbalance metric:
           ΔXZ = (rate_X - rate_Z) / (rate_X + rate_Z + ε)
       where rate_X = defects on X stabilizers per round,
             rate_Z = defects on Z stabilizers per round.
    3. Use dual-threshold hysteresis to select the schedule:
           ΔXZ > θ_enter  → switch to X_HEAVY (more X stabilizers
                            to better track dominant phase errors)
           ΔXZ < -θ_enter → switch to Z_HEAVY
           |ΔXZ| < θ_exit → revert to BALANCED
    4. Feed the selected schedule into the circuit compiler.

Hysteresis prevents boundary chattering: the entering threshold (θ_enter)
is higher than the exiting threshold (θ_exit), creating a dead zone.

References:
    - Tuckett et al., "Tailoring Surface Codes for Highly Biased Noise"
      PRX Quantum 1, 010310 (2020)
    - Bonilla Ataides et al., "The XZZX surface code" (2021)
    - Tiurev et al., "Correcting non-independent and non-identically
      distributed errors with surface codes" (2023)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from adaptive_qec.qec.schedules import (
    PREDEFINED_SCHEDULES,
    ScheduleType,
    StabilizerSchedule,
    get_schedule,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

@dataclass
class AdaptiveSchedulerConfig:
    """
    Configuration for the adaptive X/Z scheduler.

    Attributes:
        window_size: Number of recent rounds for EWMA computation.
        ewma_alpha: EWMA smoothing factor (higher = more reactive).
        theta_enter: Imbalance threshold to trigger schedule change.
        theta_exit: Imbalance threshold to revert to balanced.
        min_rounds_before_switch: Minimum rounds in current schedule
            before allowing another switch (anti-chattering guard).
        initial_schedule: Starting schedule before observations.
    """

    window_size: int = 100
    ewma_alpha: float = 0.05
    theta_enter: float = 0.15
    theta_exit: float = 0.05
    min_rounds_before_switch: int = 20
    initial_schedule: ScheduleType = ScheduleType.BALANCED

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0 < self.ewma_alpha <= 1.0:
            raise ValueError(
                f"ewma_alpha must be in (0, 1], got {self.ewma_alpha}"
            )
        if self.theta_exit >= self.theta_enter:
            raise ValueError(
                f"theta_exit ({self.theta_exit}) must be strictly less than "
                f"theta_enter ({self.theta_enter}) for hysteresis"
            )
        if self.min_rounds_before_switch < 1:
            raise ValueError(
                f"min_rounds_before_switch must be >= 1, "
                f"got {self.min_rounds_before_switch}"
            )
        if self.window_size < 10:
            raise ValueError(
                f"window_size must be >= 10, got {self.window_size}"
            )


# -----------------------------------------------------------------------
# Defect observation record
# -----------------------------------------------------------------------

@dataclass
class DefectObservation:
    """
    Syndrome defect counts from a single QEC round.

    Attributes:
        round_idx: QEC round index.
        x_defects: Number of X-type stabilizer defects detected.
        z_defects: Number of Z-type stabilizer defects detected.
        total_x_stabilizers: Total X-type stabilizers measured.
        total_z_stabilizers: Total Z-type stabilizers measured.
    """

    round_idx: int
    x_defects: int
    z_defects: int
    total_x_stabilizers: int
    total_z_stabilizers: int

    @property
    def x_defect_rate(self) -> float:
        """X defect rate normalized by stabilizer count."""
        if self.total_x_stabilizers == 0:
            return 0.0
        return self.x_defects / self.total_x_stabilizers

    @property
    def z_defect_rate(self) -> float:
        """Z defect rate normalized by stabilizer count."""
        if self.total_z_stabilizers == 0:
            return 0.0
        return self.z_defects / self.total_z_stabilizers


# -----------------------------------------------------------------------
# Adaptive scheduler
# -----------------------------------------------------------------------

class AdaptiveXZScheduler:
    """
    Online adaptive scheduler for X/Z stabilizer measurement frequency.

    Observes syndrome defect rates in real time and adjusts the
    measurement schedule to allocate more rounds to the stabilizer
    type that is detecting more errors. Uses EWMA smoothing and
    dual-threshold hysteresis to avoid boundary chattering.

    Usage:
        scheduler = AdaptiveXZScheduler(config)
        for round_idx, syndromes in enumerate(syndrome_stream):
            obs = DefectObservation(
                round_idx=round_idx,
                x_defects=count_x_defects(syndromes),
                z_defects=count_z_defects(syndromes),
                total_x_stabilizers=n_x,
                total_z_stabilizers=n_z,
            )
            schedule = scheduler.update(obs)
            # Use schedule.round_type(round_idx) for next circuit
    """

    def __init__(
        self,
        config: Optional[AdaptiveSchedulerConfig] = None,
    ) -> None:
        self._config = config or AdaptiveSchedulerConfig()
        self._config.validate()

        # State
        self._ewma_x: float = 0.0
        self._ewma_z: float = 0.0
        self._imbalance: float = 0.0
        self._current_schedule = get_schedule(self._config.initial_schedule)
        self._rounds_in_current: int = 0
        self._total_rounds: int = 0

        # History for analysis
        self._history: deque[dict[str, Any]] = deque(
            maxlen=self._config.window_size * 2
        )
        self._switch_log: list[dict[str, Any]] = []

    @property
    def current_schedule(self) -> StabilizerSchedule:
        """The currently active measurement schedule."""
        return self._current_schedule

    @property
    def imbalance(self) -> float:
        """Current syndrome imbalance metric ΔXZ ∈ [-1, 1]."""
        return self._imbalance

    @property
    def ewma_x(self) -> float:
        """Current EWMA estimate of X defect rate."""
        return self._ewma_x

    @property
    def ewma_z(self) -> float:
        """Current EWMA estimate of Z defect rate."""
        return self._ewma_z

    @property
    def switch_count(self) -> int:
        """Number of schedule switches so far."""
        return len(self._switch_log)

    def update(self, observation: DefectObservation) -> StabilizerSchedule:
        """
        Process a new defect observation and potentially update schedule.

        Args:
            observation: Defect counts from the latest QEC round.

        Returns:
            The (possibly updated) measurement schedule to use.
        """
        alpha = self._config.ewma_alpha

        # Update EWMA estimates
        self._ewma_x = (
            alpha * observation.x_defect_rate
            + (1 - alpha) * self._ewma_x
        )
        self._ewma_z = (
            alpha * observation.z_defect_rate
            + (1 - alpha) * self._ewma_z
        )

        # Compute imbalance metric ΔXZ ∈ [-1, 1]
        total_rate = self._ewma_x + self._ewma_z
        eps = 1e-10
        self._imbalance = (self._ewma_x - self._ewma_z) / (total_rate + eps)

        self._total_rounds += 1
        self._rounds_in_current += 1

        # Record history
        self._history.append({
            "round": observation.round_idx,
            "ewma_x": self._ewma_x,
            "ewma_z": self._ewma_z,
            "imbalance": self._imbalance,
            "schedule": self._current_schedule.schedule_type.value,
        })

        # Check if we can switch (anti-chattering guard)
        if self._rounds_in_current < self._config.min_rounds_before_switch:
            return self._current_schedule

        # Hysteresis decision logic
        new_schedule = self._decide_schedule()
        if new_schedule.schedule_type != self._current_schedule.schedule_type:
            self._switch_log.append({
                "round": observation.round_idx,
                "total_round": self._total_rounds,
                "from": self._current_schedule.schedule_type.value,
                "to": new_schedule.schedule_type.value,
                "imbalance": self._imbalance,
                "ewma_x": self._ewma_x,
                "ewma_z": self._ewma_z,
            })
            logger.info(
                f"Schedule switch at round {observation.round_idx}: "
                f"{self._current_schedule.schedule_type.value} → "
                f"{new_schedule.schedule_type.value} "
                f"(ΔXZ={self._imbalance:.4f})"
            )
            self._current_schedule = new_schedule
            self._rounds_in_current = 0

        return self._current_schedule

    def _decide_schedule(self) -> StabilizerSchedule:
        """
        Apply dual-threshold hysteresis to decide the schedule.

        The logic is:
          - If currently BALANCED:
              * ΔXZ > θ_enter → X_HEAVY (X defects dominating)
              * ΔXZ < -θ_enter → Z_HEAVY (Z defects dominating)
          - If currently X_HEAVY:
              * |ΔXZ| < θ_exit → BALANCED (imbalance resolved)
              * ΔXZ > 0.4 → EXTREME_X (very strong X dominance)
          - If currently Z_HEAVY:
              * |ΔXZ| < θ_exit → BALANCED
              * ΔXZ < -0.4 → EXTREME_Z

        Returns:
            The schedule to use.
        """
        theta_enter = self._config.theta_enter
        theta_exit = self._config.theta_exit
        current_type = self._current_schedule.schedule_type

        if current_type == ScheduleType.BALANCED:
            if self._imbalance > theta_enter:
                return get_schedule(ScheduleType.X_HEAVY)
            elif self._imbalance < -theta_enter:
                return get_schedule(ScheduleType.Z_HEAVY)

        elif current_type == ScheduleType.X_HEAVY:
            if abs(self._imbalance) < theta_exit:
                return get_schedule(ScheduleType.BALANCED)
            elif self._imbalance > 0.4:
                return get_schedule(ScheduleType.EXTREME_X)

        elif current_type == ScheduleType.EXTREME_X:
            if self._imbalance < theta_enter:
                return get_schedule(ScheduleType.X_HEAVY)

        elif current_type == ScheduleType.Z_HEAVY:
            if abs(self._imbalance) < theta_exit:
                return get_schedule(ScheduleType.BALANCED)
            elif self._imbalance < -0.4:
                return get_schedule(ScheduleType.EXTREME_Z)

        elif current_type == ScheduleType.EXTREME_Z:
            if self._imbalance > -theta_enter:
                return get_schedule(ScheduleType.Z_HEAVY)

        return self._current_schedule

    def reset(self) -> None:
        """Reset scheduler state to initial configuration."""
        self._ewma_x = 0.0
        self._ewma_z = 0.0
        self._imbalance = 0.0
        self._current_schedule = get_schedule(self._config.initial_schedule)
        self._rounds_in_current = 0
        self._total_rounds = 0
        self._history.clear()
        self._switch_log.clear()

    def summary(self) -> dict[str, Any]:
        """Summary of scheduler state and switch history."""
        return {
            "current_schedule": self._current_schedule.to_dict(),
            "ewma_x": round(self._ewma_x, 6),
            "ewma_z": round(self._ewma_z, 6),
            "imbalance": round(self._imbalance, 6),
            "total_rounds": self._total_rounds,
            "rounds_in_current": self._rounds_in_current,
            "total_switches": len(self._switch_log),
            "switch_log": self._switch_log,
        }

    def get_history(self) -> list[dict[str, Any]]:
        """Return recorded history for analysis."""
        return list(self._history)
