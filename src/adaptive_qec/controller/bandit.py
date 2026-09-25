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
# Shared arm definition
# ---------------------------------------------------------------------------

@dataclass
class BanditArm:
    """A single arm in the QEC bandit formulation.

    Each arm is a fixed (decoder, DD-policy) pair.
    """
    decoder: DecoderChoice
    dd_policy: DDSequenceType
    index: int

    @property
    def label(self) -> str:
        return f"{self.decoder.value}:{self.dd_policy.value}"


def build_arm_set() -> list[BanditArm]:
    """Construct the canonical arm set.

    K = |decoders| × |DD-policies| = 2 × 2 = 4 arms.
    We restrict to NONE and XY4 for DD to keep the action space small
    and the regret bounds tight.  CPMG/XY8 are only useful in extreme
    noise regimes and are handled by the burst-mitigation flag.
    """
    arms: list[BanditArm] = []
    idx = 0
    for dec in [DecoderChoice.MWPM, DecoderChoice.UNION_FIND]:
        for dd in [DDSequenceType.NONE, DDSequenceType.XY4]:
            arms.append(BanditArm(decoder=dec, dd_policy=dd, index=idx))
            idx += 1
    return arms


# ---------------------------------------------------------------------------
# Exp3 Controller
# ---------------------------------------------------------------------------

class Exp3Controller(BaseController):
    """Exp3 adversarial bandit controller.

