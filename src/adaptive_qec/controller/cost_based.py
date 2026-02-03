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
