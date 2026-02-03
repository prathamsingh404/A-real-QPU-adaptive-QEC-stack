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
