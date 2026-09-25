"""
Static baseline controller — always uses a fixed (decoder, DD) pair.

Used as the control arm in A/B experiments.  The static controller
ignores all hardware-state telemetry and never switches modes.
"""

from __future__ import annotations

from typing import Optional

from adaptive_qec.controller.base import BaseController
from adaptive_qec.controller.controller import (
    ControlAction,
    CostWeights,
    DecoderChoice,
    HardwareState,
)
from adaptive_qec.mitigation.dynamical_decoupling import DDSequenceType


class StaticController(BaseController):
    """Fixed-policy controller — never adapts.

    Parameters
    ----------
    decoder : DecoderChoice
        Fixed decoder to use for every window.
    dd_policy : DDSequenceType
        Fixed dynamical-decoupling policy for every window.
    weights : CostWeights, optional
        Cost weights (used only for telemetry cost tracking).
    """

    def __init__(
        self,
        decoder: DecoderChoice = DecoderChoice.UNION_FIND,
        dd_policy: DDSequenceType = DDSequenceType.NONE,
        weights: Optional[CostWeights] = None,
    ) -> None:
        super().__init__(weights=weights)
        self._decoder = decoder
        self._dd_policy = dd_policy

    @property
    def name(self) -> str:
        return f"static_{self._decoder.value}_{self._dd_policy.value}"

    def observe(self, state: HardwareState) -> None:
        self._current_state = state

    def decide(self) -> ControlAction:
        return ControlAction(
            decoder=self._decoder,
            dd_policy=self._dd_policy,
            burst_mitigation=False,
            request_recalibration=False,
            notes=f"static policy: {self._decoder.value}+{self._dd_policy.value}",
        )

    def update(self, reward: float) -> None:
        # Static controller ignores feedback
        if self._telemetry:
            self._telemetry[-1].cost = -reward
