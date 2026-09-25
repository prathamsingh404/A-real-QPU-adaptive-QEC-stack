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
    delta : float
        Confidence parameter.  With probability ≥ 1 − δ,
        Regret_T ≤ O(√(K T ln(K/δ))).
    horizon : int
        Estimated time horizon T for tuning η and β.
    weights : CostWeights, optional
        For telemetry cost tracking only.
    """

    def __init__(
        self,
        delta: float = 0.05,
        horizon: int = 1000,
        weights: Optional[CostWeights] = None,
    ) -> None:
        super().__init__(weights=weights)
        self._arms = build_arm_set()
        self._K = len(self._arms)
        self._delta = delta
        self._T = max(horizon, 1)

        # Tuned parameters per Auer et al. Theorem 3.3
        self._eta = math.sqrt(math.log(self._K) / (self._T * self._K))
        self._beta = math.sqrt(math.log(self._K / delta) / (self._T * self._K))

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

        return ControlAction(
            decoder=arm.decoder,
            dd_policy=arm.dd_policy,
            burst_mitigation=burst_mit,
            request_recalibration=False,
            notes=f"Exp3.P arm={arm.label} p={p[self._last_arm_idx]:.4f}",
        )

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
    window_size : int
        Maximum sliding window length W_max.
    confidence : float
        UCB confidence parameter (multiplier on √(ln(t)/n)).
    min_pulls : int
        Minimum pulls before an arm can be eliminated.
    weights : CostWeights, optional
        For telemetry cost tracking only.
    """

    def __init__(
        self,
        window_size: int = 50,
        confidence: float = 2.0,
        min_pulls: int = 5,
        weights: Optional[CostWeights] = None,
    ) -> None:
        super().__init__(weights=weights)
        self._arms = build_arm_set()
        self._K = len(self._arms)
        self._W = window_size
        self._c = confidence
        self._min_pulls = min_pulls
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
        return "da_se"

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
        if len(self._global_rewards) < 20:
            return False
        recent = self._global_rewards[-10:]
        older = self._global_rewards[-20:-10]
        diff = abs(float(np.mean(recent)) - float(np.mean(older)))
        se = float(np.std(older)) / math.sqrt(len(older)) + 1e-10
        return diff / se > 3.0  # z-score > 3

    def observe(self, state: HardwareState) -> None:
        self._current_state = state

    def decide(self) -> ControlAction:
        t = self._step + 1

        # Check for drift → reactivate all arms
        if self._detect_drift():
            logger.info("DA-SE: drift detected — reactivating all arms")
            self._active[:] = True
            self._arm_windows = [[] for _ in range(self._K)]
            self._last_drift_step = self._step

        # Among active arms, pick the one with highest UCB
        # If any active arm has < min_pulls, force-explore it
        active_indices = np.where(self._active)[0]
        if len(active_indices) == 0:
            # Safety: reactivate all
            self._active[:] = True
            active_indices = np.arange(self._K)

        under_explored = [
            i for i in active_indices
            if len(self._arm_windows[i]) < self._min_pulls
        ]

        if under_explored:
            self._last_arm_idx = int(self._rng.choice(under_explored))
        else:
            ucbs = {i: self._arm_ucb(i, t) for i in active_indices}
            self._last_arm_idx = max(ucbs, key=ucbs.get)  # type: ignore[arg-type]

        arm = self._arms[self._last_arm_idx]
        burst_mit = (
            self._current_state is not None
            and self._current_state.burst_active
        )

        return ControlAction(
            decoder=arm.decoder,
            dd_policy=arm.dd_policy,
            burst_mitigation=burst_mit,
            request_recalibration=False,
            notes=f"DA-SE arm={arm.label} active={int(self._active.sum())}/{self._K}",
        )

    def update(self, reward: float) -> None:
        # Add reward to the arm's sliding window
        w = self._arm_windows[self._last_arm_idx]
        w.append(reward)
        if len(w) > self._W:
            w.pop(0)

        self._global_rewards.append(reward)
        if len(self._global_rewards) > self._W * 2:
            self._global_rewards.pop(0)

        # Successive elimination: drop arms whose UCB < best LCB
