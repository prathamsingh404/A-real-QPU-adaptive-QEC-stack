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
