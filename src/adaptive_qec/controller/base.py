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

    def _record_action(self, action: ControlAction, latency_us: float = 0.0) -> None:
        """Record decision telemetry for this step."""
        state = self._current_state
        state_dict: dict[str, Any] = {}
        if state is not None:
            state_dict = {
                "defect_rate": getattr(state, "defect_rate", 0.0),
                "drift_magnitude": getattr(state, "drift_magnitude", 0.0),
                "drift_status": state.drift_status.value if hasattr(getattr(state, "drift_status", None), "value") else str(getattr(state, "drift_status", "")),
                "burst_active": getattr(state, "burst_active", False),
                "leakage_fraction": getattr(state, "leakage_fraction", 0.0),
                "t1_mean_us": getattr(state, "t1_mean_us", 0.0),
                "t2_mean_us": getattr(state, "t2_mean_us", 0.0),
                "p_2q": getattr(state, "p_2q", 0.0),
                "p_ro": getattr(state, "p_ro", 0.0),
                "code_distance": getattr(state, "code_distance", 3),
            }
        dec_val = action.decoder.value if hasattr(action.decoder, "value") else str(action.decoder)
        dd_val = action.dd_policy.value if hasattr(action.dd_policy, "value") else str(action.dd_policy)
        record = TelemetryRecord(
            step=self._step,
            timestamp_ns=time.perf_counter_ns(),
            hardware_state=state_dict,
            action_taken={
                "decoder": dec_val,
                "dd_policy": dd_val,
                "burst_mitigation": action.burst_mitigation,
                "request_recalibration": action.request_recalibration,
            },
            cost=0.0,  # filled in by update()
            decision_latency_us=latency_us,
            controller_name=self.name,
        )
        self._telemetry.append(record)
        self._step += 1

    def step(self, state: HardwareState) -> ControlAction:
        """Convenience: observe + decide in one call, with telemetry.

        Returns the selected ControlAction. Call update() separately
        once the window's decoding result is available.
        """
        t0 = time.perf_counter_ns()

        self.observe(state)
        len_before = len(self._telemetry)
        action = self.decide()

        t1 = time.perf_counter_ns()
        latency_us = (t1 - t0) / 1_000.0

        self._current_state = state
        self._current_action = action

        if len(self._telemetry) > len_before:
            self._telemetry[-1].decision_latency_us = latency_us
        else:
            self._record_action(action, latency_us)

        return action


    def reset(self) -> None:
        """Reset controller state for a new experiment run."""
        self._step = 0
        self._telemetry.clear()
        self._current_state = None
        self._current_action = None

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable controller summary."""
        latencies = [r.decision_latency_us for r in self._telemetry]
        costs = [r.cost for r in self._telemetry if r.cost != 0.0]
        return {
            "controller": self.name,
            "total_steps": self._step,
            "mean_decision_latency_us": float(np.mean(latencies)) if latencies else 0.0,
            "p99_decision_latency_us": float(np.percentile(latencies, 99)) if latencies else 0.0,
            "mean_cost": float(np.mean(costs)) if costs else 0.0,
        }

    @property
    def telemetry(self) -> list[TelemetryRecord]:
        """Access the telemetry stream for offline analysis."""
        return self._telemetry
