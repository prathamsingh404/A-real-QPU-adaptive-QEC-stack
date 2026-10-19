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

    def to_dict(self) -> dict[str, Any]:
        return {
            "p50_us": round(self.p50_us, 2),
            "p90_us": round(self.p90_us, 2),
            "p95_us": round(self.p95_us, 2),
            "p99_us": round(self.p99_us, 2),
            "mean_us": round(self.mean_us, 2),
            "max_us": round(self.max_us, 2),
            "throughput_shots_per_s": round(self.throughput_shots_per_s, 2),
            "breakdown_mean": self.breakdown_mean.to_dict(),
            "timescale_classification": self.timescale_classification,
        }


class LatencyProfiler:
    """Profiles latency distributions across measurement, decoding, decision, and communication."""

    def __init__(self) -> None:
        self.recorded_totals: list[float] = []
        self.recorded_breakdowns: list[LatencyBreakdown] = []

    def record(
        self,
        t_meas_us: float,
        t_decode_us: float,
        t_decision_us: float,
        t_comm_us: float = 0.5,
    ) -> None:
        lb = LatencyBreakdown(
            t_meas_us=t_meas_us,
            t_decode_us=t_decode_us,
            t_decision_us=t_decision_us,
            t_comm_us=t_comm_us,
        )
        self.recorded_breakdowns.append(lb)
        self.recorded_totals.append(lb.total_us)

    def summarize(self) -> LatencyProfileResult:
        if not self.recorded_totals:
            zero_bd = LatencyBreakdown(0, 0, 0, 0)
            return LatencyProfileResult(0, 0, 0, 0, 0, 0, 0, zero_bd, "unprofiled")

        totals = np.array(self.recorded_totals)
        mean_meas = float(np.mean([b.t_meas_us for b in self.recorded_breakdowns]))
        mean_dec = float(np.mean([b.t_decode_us for b in self.recorded_breakdowns]))
        mean_ctrl = float(np.mean([b.t_decision_us for b in self.recorded_breakdowns]))
        mean_comm = float(np.mean([b.t_comm_us for b in self.recorded_breakdowns]))

        mean_total = float(np.mean(totals))
        throughput = 1e6 / max(mean_total, 1e-6)

        # Classification
        classification = (
            "microsecond_feedforward"
            if mean_total < 10.0
            else "macro_orchestration"
        )

        return LatencyProfileResult(
            p50_us=float(np.percentile(totals, 50)),
            p90_us=float(np.percentile(totals, 90)),
            p95_us=float(np.percentile(totals, 95)),
            p99_us=float(np.percentile(totals, 99)),
            mean_us=mean_total,
            max_us=float(np.max(totals)),
            throughput_shots_per_s=throughput,
            breakdown_mean=LatencyBreakdown(mean_meas, mean_dec, mean_ctrl, mean_comm),
            timescale_classification=classification,
        )


@dataclass
class FeasibilityPoint:
    tau_drift_s: float
    tau_adapt_s: float
    timescale_ratio_rho: float
    is_feasible: bool
    net_benefit_fraction: float
    recommendation: str  # "ADAPT" or "STAY"


def compute_adaptation_feasibility(
    tau_drift_s: float,
    tau_adapt_s: float,
    base_benefit: float = 0.08,
) -> FeasibilityPoint:
    """
    Computes timescale ratio rho = tau_adapt / tau_drift and net benefit.

    When rho > 0.5, adaptation lag prevents capitalizing on drift before noise shifts.
    """
    rho = tau_adapt_s / max(tau_drift_s, 1e-6)

    # Net benefit decays exponentially with timescale ratio
    # When rho is very small, we capture 100% of base benefit minus tax
    net_benefit = base_benefit * np.exp(-1.5 * rho) - (0.02 * rho)

    is_feasible = rho < 0.3 and net_benefit > 0.005
    recommendation = "ADAPT" if is_feasible else "STAY"

    return FeasibilityPoint(
        tau_drift_s=tau_drift_s,
        tau_adapt_s=tau_adapt_s,
        timescale_ratio_rho=float(rho),
        is_feasible=is_feasible,
        net_benefit_fraction=float(net_benefit),
        recommendation=recommendation,
    )
