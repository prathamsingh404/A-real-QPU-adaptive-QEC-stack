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
    ) -> None:
        super().__init__(weights=weights)
        self._gamma = gamma
        self._arms = arms if arms is not None else build_arm_set()
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
        dec_val = arm.decoder.value if hasattr(arm.decoder, "value") else str(arm.decoder)
        dd_val = getattr(arm, "dd_sequence", None) or (arm.dd_policy.value if hasattr(arm.dd_policy, "value") else str(arm.dd_policy))
        sched_val = getattr(arm, "schedule", "balanced")

        action = ControlAction(
            decoder=dec_val,
            dd_policy=arm.dd_policy,
            burst_mitigation=burst_mit,
            request_recalibration=False,
            dd_sequence=dd_val,
            schedule=sched_val,
            notes=f"Exp3 arm={arm.label} p={p[self._last_arm_idx]:.4f}",
        )
        self._record_action(action)
        return action


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

        # Prevent overflow by recentering
        self._log_weights -= self._log_weights.max()

        # Track statistics
        self._arm_pull_counts[self._last_arm_idx] += 1
        self._arm_reward_sums[self._last_arm_idx] += reward

        if self._telemetry:
            self._telemetry[-1].cost = -reward
            self._telemetry[-1].extras["arm_idx"] = self._last_arm_idx
            self._telemetry[-1].extras["probs"] = p.tolist()

    def reset(self) -> None:
        super().reset()
        self._log_weights = np.zeros(self._K, dtype=np.float64)
        self._arm_pull_counts = np.zeros(self._K, dtype=np.int64)
        self._arm_reward_sums = np.zeros(self._K, dtype=np.float64)

    def summary(self) -> dict[str, Any]:
        base = super().summary()
        base["gamma"] = self._gamma
        base["arm_labels"] = [a.label for a in self._arms]
        base["arm_pull_counts"] = self._arm_pull_counts.tolist()
        base["arm_mean_rewards"] = [
            float(self._arm_reward_sums[i] / max(self._arm_pull_counts[i], 1))
            for i in range(self._K)
        ]
        base["current_probs"] = self._compute_probs().tolist()
        return base


# ---------------------------------------------------------------------------
# Exp3.P Controller
# ---------------------------------------------------------------------------

class Exp3PController(BaseController):
    """Exp3.P adversarial bandit with explicit exploration bonus.

    Adds an exploration bonus b_i,t to each arm's cumulative reward
    estimate, guaranteeing a *high-probability* regret bound instead
    of an in-expectation bound.

