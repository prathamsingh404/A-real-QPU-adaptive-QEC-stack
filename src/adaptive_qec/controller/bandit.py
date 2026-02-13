"""
Bandit-based adaptive controllers for QEC policy selection.

Implements three adversarial multi-armed bandit algorithms over the
finite action space {(decoder, DD-policy)} ∈ {MWPM, UF} × {NONE, XY4}:

    1. Exp3 — Exponential-weight for Exploration and Exploitation.
    2. Exp3.P — Exp3 with explicit exploration bonus for high-probability
       regret bounds.  Formally: O(√(K T ln K)) regret.
    3. DA-SE (Drift-Aware Successive Elimination) — a non-stationary
       extension that maintains per-arm sliding-window means and
       eliminates arms whose confidence bounds are dominated.

All algorithms operate on *real observed rewards* from hardware
execution, not simulated costs.

References:
    - Auer et al., "The Nonstochastic Multiarmed Bandit Problem" (2002)
    - Besbes, Gur, Zeevi, "Non-Stationary Stochastic Optimization" (2015)
    - Roadmap: PROJECT_EVOLUTION_ROADMAP.md §Phase-2
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from adaptive_qec.controller.base import BaseController
from adaptive_qec.controller.controller import (
    ControlAction,
    CostWeights,
    DecoderChoice,
    HardwareState,
)
from adaptive_qec.mitigation.dynamical_decoupling import DDSequenceType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Shared arm definition
# ---------------------------------------------------------------------------

@dataclass
class BanditArm:
    """A single arm in the QEC bandit formulation.

    Each arm is a fixed (decoder, DD-policy, schedule) tuple.
    """
    decoder: Any
    dd_policy: Any = DDSequenceType.NONE
    index: int = 0
    dd_sequence: Optional[str] = None
    schedule: str = "balanced"

    def __post_init__(self) -> None:
        if self.dd_sequence is None:
            val = self.dd_policy.value if hasattr(self.dd_policy, "value") else str(self.dd_policy)
            object.__setattr__(self, "dd_sequence", val)
        elif self.dd_policy == DDSequenceType.NONE and self.dd_sequence != "none":
            dd_map = {
                "none": DDSequenceType.NONE,
                "xy4": DDSequenceType.XY4,
                "xy8": DDSequenceType.XY8,
                "cpmg": DDSequenceType.CPMG,
            }
            if str(self.dd_sequence).lower() in dd_map:
                object.__setattr__(self, "dd_policy", dd_map[str(self.dd_sequence).lower()])

    @property
    def label(self) -> str:
        dec = self.decoder.value if hasattr(self.decoder, "value") else str(self.decoder)
        dd = self.dd_sequence or (self.dd_policy.value if hasattr(self.dd_policy, "value") else str(self.dd_policy))
        return f"{dec}:{dd}"


def build_arm_set() -> list[BanditArm]:
    """Construct the standard 6-arm set: {MWPM, UF} x {NONE, XY4, XY8}."""
    arms: list[BanditArm] = []
    idx = 0
    for dec in ["mwpm", "union_find"]:
        for dd in ["none", "xy4", "xy8"]:
            arms.append(BanditArm(
                decoder=dec,
                dd_policy=DDSequenceType(dd) if dd in [e.value for e in DDSequenceType] else DDSequenceType.NONE,
                index=idx,
                dd_sequence=dd,
                schedule="balanced",
            ))
            idx += 1
    return arms


# ---------------------------------------------------------------------------
# Exp3 Controller
# ---------------------------------------------------------------------------

class Exp3Controller(BaseController):
    """Exp3 adversarial bandit controller.

    Maintains a probability distribution p_t over K arms using
    exponential weights.  At each step, samples an arm from p_t,
    observes the reward, and updates weights.
    """

    def __init__(
        self,
        arms: Optional[list[BanditArm]] = None,
        gamma: float = 0.15,
        weights: Optional[CostWeights] = None,
        **kwargs: Any,
