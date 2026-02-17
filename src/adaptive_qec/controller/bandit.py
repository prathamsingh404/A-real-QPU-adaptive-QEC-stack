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

    Parameters
    ----------
    arms : list[BanditArm], optional
        Candidate arm set. Defaults to build_arm_set().
    gamma : float, optional
        Exploration mixing parameter.
    beta : float, optional
        Bonus parameter. If None, tuned according to Auer et al.
    delta : float
        Confidence parameter. With probability ≥ 1 − δ,
        Regret_T ≤ O(√(K T ln(K/δ))).
    horizon : int
        Estimated time horizon T for tuning η and β.
    weights : CostWeights, optional
        For telemetry cost tracking only.
    """

    def __init__(
        self,
        arms: Optional[list[BanditArm]] = None,
        gamma: float = 0.1,
        beta: Optional[float] = None,
        delta: float = 0.05,
        horizon: int = 1000,
        weights: Optional[CostWeights] = None,
    ) -> None:
        super().__init__(weights=weights)
        self._arms = arms if arms is not None else build_arm_set()
        self._K = len(self._arms)
        self._gamma = gamma
        self._delta = delta
        self._T = max(horizon, 1)

        # Tuned parameters per Auer et al. Theorem 3.3
        self._eta = gamma if gamma is not None else math.sqrt(math.log(self._K) / (self._T * self._K))
        self._beta = beta if beta is not None else math.sqrt(math.log(self._K / delta) / (self._T * self._K))

        self._log_weights = np.zeros(self._K, dtype=np.float64)
        self._last_arm_idx: int = 0
        self._rng = np.random.default_rng()

        # Counters
        self._arm_pull_counts = np.zeros(self._K, dtype=np.int64)
        self._arm_reward_sums = np.zeros(self._K, dtype=np.float64)

    @property
    def name(self) -> str:
        return "exp3p"

    def _compute_probs(self) -> np.ndarray:
        shifted = self._log_weights - self._log_weights.max()
        exp_w = np.exp(shifted)
        raw = exp_w / exp_w.sum()
        # Mix with uniform: α = η
        alpha = min(self._eta, 1.0 / self._K)
        p = (1.0 - self._K * alpha) * raw + alpha
        p = np.clip(p, 1e-12, None)
        p /= p.sum()
        return p

    def observe(self, state: HardwareState) -> None:
        self._current_state = state

    def decide(self) -> ControlAction:
        p = self._compute_probs()
        self._last_arm_idx = int(self._rng.choice(self._K, p=p))
        arm = self._arms[self._last_arm_idx]

        burst_mit = (
            self._current_state is not None
            and self._current_state.burst_active
        )

        dd_seq = getattr(arm, "dd_sequence", arm.dd_policy.value if hasattr(arm.dd_policy, "value") else str(arm.dd_policy))
        sched = getattr(arm, "schedule", "balanced")
        dec_str = arm.decoder.value if hasattr(arm.decoder, "value") else str(arm.decoder)

        action = ControlAction(
            decoder=arm.decoder,
            dd_policy=arm.dd_policy,
            dd_sequence=dd_seq,
            schedule=sched,
            burst_mitigation=burst_mit,
            request_recalibration=False,
            notes=f"Exp3.P arm={arm.label} p={p[self._last_arm_idx]:.4f}",
        )
        self._record_action(action)
        return action


    def update(self, reward: float) -> None:
        p = self._compute_probs()
        p_i = p[self._last_arm_idx]

        # Importance-weighted estimate + exploration bonus for ALL arms
        for i in range(self._K):
            bonus = self._beta / p[i]
            if i == self._last_arm_idx:
                r_hat = reward / p_i + bonus
            else:
                r_hat = bonus
            self._log_weights[i] += self._eta * r_hat

        # Recenter
        self._log_weights -= self._log_weights.max()

        self._arm_pull_counts[self._last_arm_idx] += 1
        self._arm_reward_sums[self._last_arm_idx] += reward

        if self._telemetry:
            self._telemetry[-1].cost = -reward
            self._telemetry[-1].extras["arm_idx"] = self._last_arm_idx

    def reset(self) -> None:
        super().reset()
        self._log_weights = np.zeros(self._K, dtype=np.float64)
        self._arm_pull_counts = np.zeros(self._K, dtype=np.int64)
        self._arm_reward_sums = np.zeros(self._K, dtype=np.float64)

    def summary(self) -> dict[str, Any]:
        base = super().summary()
        base["eta"] = self._eta
        base["beta"] = self._beta
        base["delta"] = self._delta
        base["arm_labels"] = [a.label for a in self._arms]
        base["arm_pull_counts"] = self._arm_pull_counts.tolist()
        base["current_probs"] = self._compute_probs().tolist()
        return base



# ---------------------------------------------------------------------------
# DA-SE (Drift-Aware Successive Elimination)
# ---------------------------------------------------------------------------

class DASEController(BaseController):
    """Drift-Aware Successive Elimination bandit controller.

    For non-stationary environments (drifting QPU noise), maintains
    per-arm sliding-window reward statistics and eliminates arms
    whose upper confidence bound is below the best arm's lower bound.

    The sliding window length W adapts to the estimated drift rate:
        W = min(W_max, ceil(1 / estimated_drift_rate))

    Eliminated arms are reactivated when drift is detected (change-point),
    implementing a "restart on drift" policy.

    Parameters
    ----------
    arms : list[BanditArm], optional
        Candidate arm set. Defaults to build_arm_set().
    window_size : int
        Maximum sliding window length W_max.
    confidence : float
        UCB confidence parameter (multiplier on √(ln(t)/n)).
    exploration_bonus : float, optional
        Alias for confidence multiplier.
    min_pulls : int
        Minimum pulls before an arm can be eliminated.
    drift_window : int, optional
        Window size for drift detection.
    drift_threshold : float, optional
        Threshold for drift detection.
    weights : CostWeights, optional
        For telemetry cost tracking only.
    """

    def __init__(
        self,
        arms: Optional[list[BanditArm]] = None,
        window_size: int = 50,
        confidence: float = 2.0,
        exploration_bonus: Optional[float] = None,
        min_pulls: int = 5,
        drift_window: Optional[int] = None,
        drift_threshold: float = 0.01,
        weights: Optional[CostWeights] = None,
    ) -> None:
        super().__init__(weights=weights)
        self._arms = arms if arms is not None else build_arm_set()
        self._K = len(self._arms)
        self._W = window_size
        self._c = exploration_bonus if exploration_bonus is not None else confidence
        self._min_pulls = min_pulls
        self._drift_window = drift_window if drift_window is not None else 20
        self._drift_threshold = drift_threshold
        self._rng = np.random.default_rng()

        # Per-arm sliding window of recent rewards
        self._arm_windows: list[list[float]] = [[] for _ in range(self._K)]
        self._active: np.ndarray = np.ones(self._K, dtype=bool)
        self._last_arm_idx: int = 0

        # Drift detection: track global reward moving average
        self._global_rewards: list[float] = []
        self._last_drift_step: int = 0

    @property
    def name(self) -> str:
        return "dase"

    def _arm_mean(self, i: int) -> float:
        w = self._arm_windows[i]
        return float(np.mean(w)) if w else 0.0

    def _arm_ucb(self, i: int, t: int) -> float:
        w = self._arm_windows[i]
        n = len(w)
        if n == 0:
            return float("inf")
        mean = float(np.mean(w))
        bonus = self._c * math.sqrt(math.log(max(t, 2)) / n)
        return mean + bonus

    def _arm_lcb(self, i: int, t: int) -> float:
        w = self._arm_windows[i]
        n = len(w)
        if n == 0:
            return float("-inf")
        mean = float(np.mean(w))
        bonus = self._c * math.sqrt(math.log(max(t, 2)) / n)
        return mean - bonus

    def _detect_drift(self) -> bool:
        """Simple CUSUM-like drift detector on global reward stream."""
        dw = self._drift_window
        if len(self._global_rewards) < dw:
            return False
        half = max(dw // 2, 2)
        recent = self._global_rewards[-half:]
