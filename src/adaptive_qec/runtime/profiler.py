"""
Runtime profiler — end-to-end timing breakdown.

Records:
    QPU measurement → transport → syndrome conversion →
    ML inference → decoder → decision

Calculates:
    T_total = T_acquisition + T_transport + T_preprocess +
              T_inference + T_decode + T_return

Reports:
    P50, P95, P99, P999 latency histograms.

Tracks latency budget:
    deadline / actual / slack → SAFE / WARNING / MISSED
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class LatencyClassification(str, Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    MISSED = "MISSED"


@dataclass
class StageTimings:
    """Timing data for a single pipeline stage."""
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_s: float = 0.0


@dataclass
class LatencyBudgetStatus:
    """Status of the latency budget for a QEC round."""
    deadline_us: float
    actual_us: float
    slack_us: float
    classification: LatencyClassification


class PipelineProfiler:
    """
    End-to-end pipeline profiler.

    Don't just measure Python runtime — measure each stage of the
    QEC decoding pipeline.
    """

    def __init__(self) -> None:
        self._stages: dict[str, StageTimings] = {}
        self._stage_order: list[str] = []
        self._round_latencies: list[float] = []

    def start_stage(self, name: str) -> None:
        """Mark the start of a pipeline stage."""
        timing = StageTimings(name=name, start_time=time.perf_counter())
        self._stages[name] = timing
        if name not in self._stage_order:
            self._stage_order.append(name)

    def end_stage(self, name: str) -> float:
        """
        Mark the end of a pipeline stage.

        Returns:
            Duration in seconds.
        """
        if name not in self._stages:
            logger.warning(f"Stage '{name}' was not started")
            return 0.0

        timing = self._stages[name]
        timing.end_time = time.perf_counter()
        timing.duration_s = timing.end_time - timing.start_time

        logger.debug(f"Stage '{name}': {timing.duration_s*1000:.2f}ms")
        return timing.duration_s

    def record_round_latency(self, latency_us: float) -> None:
        """Record the latency of a single QEC round decode."""
        self._round_latencies.append(latency_us)

    def check_latency_budget(
        self,
        deadline_us: float,
        actual_us: float,
        warning_threshold: float = 0.8,
    ) -> LatencyBudgetStatus:
        """
        Check whether a decode met the latency budget.

        Args:
            deadline_us: latency budget in microseconds.
            actual_us: actual decode time in microseconds.
            warning_threshold: fraction of deadline for WARNING.

        Returns:
            LatencyBudgetStatus with classification.
        """
        slack = deadline_us - actual_us

        if actual_us > deadline_us:
            classification = LatencyClassification.MISSED
        elif actual_us > deadline_us * warning_threshold:
            classification = LatencyClassification.WARNING
        else:
            classification = LatencyClassification.SAFE

        return LatencyBudgetStatus(
            deadline_us=deadline_us,
            actual_us=actual_us,
            slack_us=slack,
            classification=classification,
        )

    def get_report(self) -> dict:
        """
        Generate a complete timing report.

        Returns:
