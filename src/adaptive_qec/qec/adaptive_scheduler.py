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
