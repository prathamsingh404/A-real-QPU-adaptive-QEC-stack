"""
Cost-based adaptive controller wrapped as a BaseController.

This wraps the existing `AdaptiveController` from controller.py
so it can participate in the experiment harness alongside bandits
and static baselines.
"""

from __future__ import annotations

from typing import Optional

from adaptive_qec.controller.base import BaseController
from adaptive_qec.controller.controller import (
    AdaptiveController,
    ControlAction,
    CostWeights,
    HardwareState,
)


class CostBasedController(BaseController):
    """Wraps the existing argmin-cost adaptive controller.

    This is the deterministic cost-enumeration policy that evaluates
    J(a|s) for every candidate action and picks the minimum subject
    to hysteresis.  It serves as the "adaptive but not bandit" arm.

    Parameters
    ----------
    weights : CostWeights, optional
        Weights for the multi-objective cost J(a|s).
    hysteresis_patience : int
        Consecutive windows the candidate must dominate before switch.
    hysteresis_margin : float
        Minimum fractional improvement to count as a "win".
    """

    def __init__(
        self,
        weights: Optional[CostWeights] = None,
        hysteresis_patience: int = 3,
        hysteresis_margin: float = 0.05,
    ) -> None:
        super().__init__(weights=weights)
        self._inner = AdaptiveController(
            weights=self._weights,
            hysteresis_patience=hysteresis_patience,
            hysteresis_margin=hysteresis_margin,
        )
        self._last_action: Optional[ControlAction] = None

    @property
    def name(self) -> str:
        return "cost_based_adaptive"

    def observe(self, state: HardwareState) -> None:
        self._current_state = state

    def decide(self) -> ControlAction:
        if self._current_state is None:
            raise RuntimeError("observe() must be called before decide()")
        action = self._inner.select_action(self._current_state)
        self._last_action = action
        return action

    def update(self, reward: float) -> None:
        if self._telemetry:
            self._telemetry[-1].cost = -reward

    def reset(self) -> None:
        super().reset()
        self._inner.reset()
        self._last_action = None

    def summary(self) -> dict:
        base = super().summary()
        base["inner_metrics"] = self._inner.summary()
        return base
