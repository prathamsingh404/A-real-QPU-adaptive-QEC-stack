"""
Abstract controller interface for QEC policy selection.

Every controller strategy (static baseline, cost-based, bandit,
SPRT-augmented) implements this interface. The experiment harness
calls `observe() → decide() → update()` once per observation window.

Design decision: controllers are *stateful* (they accumulate history
for hysteresis, bandit arms, etc.) but *interpretable* (every decision
is traceable via the telemetry stream).
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from adaptive_qec.controller.controller import (
    ControlAction,
    ControllerMetrics,
    CostWeights,
    HardwareState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Telemetry record — one per decision step
# ---------------------------------------------------------------------------

@dataclass
class TelemetryRecord:
    """Immutable record of a single controller decision.

    Stored for offline analysis, regret computation, and reproducibility.
    """
    step: int
    timestamp_ns: int
    hardware_state: dict[str, Any]
    action_taken: dict[str, Any]
    cost: float
    decision_latency_us: float
    controller_name: str
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "timestamp_ns": self.timestamp_ns,
            "hardware_state": self.hardware_state,
            "action_taken": self.action_taken,
            "cost": self.cost,
            "decision_latency_us": self.decision_latency_us,
            "controller_name": self.controller_name,
            "extras": self.extras,
        }


# ---------------------------------------------------------------------------
# Abstract base controller
# ---------------------------------------------------------------------------

class BaseController(ABC):
    """Abstract interface for all QEC control policies.
