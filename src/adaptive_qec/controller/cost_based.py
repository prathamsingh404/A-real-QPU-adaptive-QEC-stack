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
