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

    Maintains a probability distribution p_t over K arms using
    exponential weights.  At each step, samples an arm from p_t,
    observes the reward, and updates weights.

    Parameters
    ----------
    gamma : float
        Exploration-exploitation mixing parameter in (0, 1].
        Higher → more uniform exploration.  Default is tuned for
        K=4 arms and T~1000 windows.
    weights : CostWeights, optional
        For telemetry cost tracking only.

    Formal guarantee:
        E[Regret_T] ≤ 2 √(K T ln K)  when γ = √(K ln K / T).
    """

    def __init__(
        self,
        gamma: float = 0.15,
        weights: Optional[CostWeights] = None,
    ) -> None:
        super().__init__(weights=weights)
        self._gamma = gamma
        self._arms = build_arm_set()
        self._K = len(self._arms)
        self._log_weights = np.zeros(self._K, dtype=np.float64)
        self._last_arm_idx: int = 0
        self._rng = np.random.default_rng()

        # History for analysis
        self._arm_pull_counts = np.zeros(self._K, dtype=np.int64)
        self._arm_reward_sums = np.zeros(self._K, dtype=np.float64)

    @property
    def name(self) -> str:
        return "exp3"

    def _compute_probs(self) -> np.ndarray:
        """Compute the mixed strategy p_t from log-weights."""
        shifted = self._log_weights - self._log_weights.max()
        exp_w = np.exp(shifted)
        raw = exp_w / exp_w.sum()
        # mix with uniform
        p = (1.0 - self._gamma) * raw + self._gamma / self._K
        p = np.clip(p, 1e-12, None)
        p /= p.sum()
        return p

    def observe(self, state: HardwareState) -> None:
        self._current_state = state

    def decide(self) -> ControlAction:
        p = self._compute_probs()
        self._last_arm_idx = int(self._rng.choice(self._K, p=p))
        arm = self._arms[self._last_arm_idx]

        # Burst mitigation: always enable if burst detected
        burst_mit = (
            self._current_state is not None
            and self._current_state.burst_active
        )

        return ControlAction(
            decoder=arm.decoder,
            dd_policy=arm.dd_policy,
            burst_mitigation=burst_mit,
            request_recalibration=False,
            notes=f"Exp3 arm={arm.label} p={p[self._last_arm_idx]:.4f}",
        )

    def update(self, reward: float) -> None:
        """Update weights with importance-weighted reward estimate.

        reward ∈ [0, 1] is the survival probability (1 - logical_error_rate).
        """
        p = self._compute_probs()
        p_i = p[self._last_arm_idx]

        # Importance-weighted reward estimate
        r_hat = reward / p_i

        # Update log-weight
        eta = self._gamma / self._K
        self._log_weights[self._last_arm_idx] += eta * r_hat

