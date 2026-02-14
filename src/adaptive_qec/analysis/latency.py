"""
Latency Stack Profiling, Timescales & Adaptation Feasibility Map (TASK-057 to TASK-062).

Models and profiles the complete QEC latency stack:
    T_total = T_meas + T_decode + T_decision + T_comm

Formalizes the drift-to-controller timescale ratio:
    rho = tau_adapt / tau_drift

And defines the Adaptation Feasibility Map:
    - rho << 1: Feasible adaptation regime (confident gain > tax)
    - rho ~ 1: Critical transition zone (requires strict dead-zone gating)
    - rho >> 1: Infeasible regime (noise drifts faster than evidence accumulation -> STAY)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LatencyBreakdown:
    """Breakdown of latency components in microseconds."""
    t_meas_us: float        # Measurement extraction time
    t_decode_us: float      # Decoder solve time
    t_decision_us: float    # Controller policy decision time
    t_comm_us: float        # Classical communication / payload serialization time

    @property
    def total_us(self) -> float:
        return self.t_meas_us + self.t_decode_us + self.t_decision_us + self.t_comm_us

    def to_dict(self) -> dict[str, float]:
        return {
            "t_meas_us": self.t_meas_us,
            "t_decode_us": self.t_decode_us,
            "t_decision_us": self.t_decision_us,
            "t_comm_us": self.t_comm_us,
            "total_us": self.total_us,
        }


@dataclass
class LatencyProfileResult:
    """Statistical distribution of latency stack across shots."""
    p50_us: float
    p90_us: float
    p95_us: float
    p99_us: float
    mean_us: float
    max_us: float
    throughput_shots_per_s: float
    breakdown_mean: LatencyBreakdown
    timescale_classification: str  # "microsecond_feedforward" or "macro_orchestration"
