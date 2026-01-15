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
