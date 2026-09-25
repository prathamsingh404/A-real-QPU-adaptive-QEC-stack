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

    Lifecycle per window:
        1. harness calls `observe(state)` — controller ingests telemetry.
        2. harness calls `decide()` — controller returns a ControlAction.
        3. after decoding, harness calls `update(reward)` — controller
           learns from the outcome (no-op for static).

    Subclasses MUST implement:
        - name (property)
        - observe(state)
        - decide()
        - update(reward)

    Subclasses MAY override:
        - reset()
        - summary()
    """

    def __init__(self, weights: Optional[CostWeights] = None) -> None:
        self._weights = weights or CostWeights()
        self._step: int = 0
        self._telemetry: list[TelemetryRecord] = []
        self._current_state: Optional[HardwareState] = None
        self._current_action: Optional[ControlAction] = None

    # -- abstract interface --------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'static', 'bandit_exp3', 'sprt'."""
        ...

    @abstractmethod
    def observe(self, state: HardwareState) -> None:
        """Ingest the current hardware-state observation.

        Called once per window *before* decide().
        """
        ...

    @abstractmethod
    def decide(self) -> ControlAction:
        """Return the control action for this window.

        Must be called *after* observe().
        """
        ...

    @abstractmethod
    def update(self, reward: float) -> None:
        """Receive a scalar reward/loss signal from the completed window.

        `reward` is typically the *negative* decoded logical error rate
        for the window, so higher is better.  Static controllers ignore
        this; bandit controllers use it to update arm weights.
        """
        ...

    # -- lifecycle -----------------------------------------------------------

    def step(self, state: HardwareState) -> ControlAction:
        """Convenience: observe + decide in one call, with telemetry.

        Returns the selected ControlAction.  Call update() separately
        once the window's decoding result is available.
        """
        t0 = time.perf_counter_ns()

        self.observe(state)
