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


