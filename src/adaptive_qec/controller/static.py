"""
Static baseline controller — always uses a fixed (decoder, DD) pair.

Used as the control arm in A/B experiments.  The static controller
ignores all hardware-state telemetry and never switches modes.
"""

from __future__ import annotations

from typing import Any, Optional

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
    decoder : DecoderChoice or str
        Fixed decoder to use for every window.
    dd_policy : DDSequenceType or str
        Fixed dynamical-decoupling policy for every window.
    dd_sequence : str, optional
        Alias for dd_policy.
    schedule : str, optional
        Stabilizer schedule type.
    weights : CostWeights, optional
        Cost weights (used only for telemetry cost tracking).
    """

    def __init__(
        self,
        decoder: Any = DecoderChoice.UNION_FIND,
        dd_policy: Any = DDSequenceType.NONE,
        dd_sequence: Optional[Any] = None,
        schedule: str = "balanced",
        weights: Optional[CostWeights] = None,
    ) -> None:
        super().__init__(weights=weights)
        if isinstance(decoder, str):
            decoder = DecoderChoice.MWPM if "mwpm" in decoder.lower() else DecoderChoice.UNION_FIND
        self._decoder = decoder

        effective_dd = dd_sequence if dd_sequence is not None else dd_policy
        if isinstance(effective_dd, str):
            dd_lower = effective_dd.lower()
            if "xy4" in dd_lower:
                effective_dd = DDSequenceType.XY4
            elif "xy8" in dd_lower:
                effective_dd = DDSequenceType.XY8
            elif "cpmg" in dd_lower:
                effective_dd = DDSequenceType.CPMG
            else:
                effective_dd = DDSequenceType.NONE
        self._dd_policy = effective_dd
        self._schedule = schedule

    @property
    def name(self) -> str:
        dec_str = self._decoder.value if hasattr(self._decoder, "value") else str(self._decoder)
        dd_str = self._dd_policy.value if hasattr(self._dd_policy, "value") else str(self._dd_policy)
        return f"static_{dec_str}_{dd_str}"

    def observe(self, state: HardwareState) -> None:
        self._current_state = state

    def decide(self) -> ControlAction:
        dec_val = self._decoder.value if hasattr(self._decoder, "value") else str(self._decoder)
        dd_val = self._dd_policy.value if hasattr(self._dd_policy, "value") else str(self._dd_policy)
        action = ControlAction(
            decoder=self._decoder,
            dd_policy=self._dd_policy,
            dd_sequence=dd_val,
            schedule=self._schedule,
            burst_mitigation=False,
            request_recalibration=False,
            notes=f"static policy: {dec_val}+{dd_val}",
        )
        self._record_action(action)
        return action

    def update(self, reward: float) -> None:
        # Static controller ignores feedback
        if self._telemetry:
            self._telemetry[-1].cost = -reward
