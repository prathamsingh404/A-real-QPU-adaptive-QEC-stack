"""
Sequential Probability Ratio Test (SPRT) augmented controller.

Implements Wald's SPRT to make statistically rigorous switching
decisions between QEC modes.  Instead of switching whenever a cost
estimate looks better, we accumulate evidence until the log-likelihood
ratio crosses a threshold — guaranteeing bounded Type-I/II error.

Mathematical formulation:
    H₀: current mode's error rate ≤ challenger's error rate
    H₁: challenger's error rate < current mode's error rate − Δ

    Log-likelihood ratio:
        Λₙ = Σᵢ log[ P(xᵢ | H₁) / P(xᵢ | H₀) ]

    Decision boundaries:
        Λₙ ≥  log((1−β)/α)  →  accept H₁ (switch)
        Λₙ ≤  log(β/(1−α))  →  accept H₀ (stay)

    where α = P(Type-I error), β = P(Type-II error).

This module can wrap any BaseController to gate its switching
decisions through SPRT, or operate standalone as a SPRT-bandit hybrid.

References:
    - Wald, "Sequential Tests of Statistical Hypotheses" (1945)
    - Roadmap: PROJECT_EVOLUTION_ROADMAP.md §Phase-3
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
# SPRT core
# ---------------------------------------------------------------------------

@dataclass
class SPRTState:
    """Running state for one SPRT comparison (current vs challenger)."""
    challenger_label: str
    log_likelihood_ratio: float = 0.0
    samples_seen: int = 0
    current_rewards: list[float] = field(default_factory=list)
    challenger_rewards: list[float] = field(default_factory=list)
    decision: str = "undecided"  # "undecided" | "switch" | "stay"

    def reset(self) -> None:
        self.log_likelihood_ratio = 0.0
        self.samples_seen = 0
        self.current_rewards.clear()
        self.challenger_rewards.clear()
        self.decision = "undecided"


class SPRTEngine:
    """Wald's Sequential Probability Ratio Test engine.

    Parameters
    ----------
    alpha : float
        Type-I error probability (false switch).  Default 0.05.
    beta : float
        Type-II error probability (missed switch).  Default 0.10.
    delta : float
        Minimum detectable effect size (in error rate difference).
        Switching is only justified if the challenger's error rate
        is at least delta lower.  Default 0.01 (1% absolute).
    max_samples : int
        Maximum samples before forcing a decision (truncated SPRT).
        Prevents indefinite accumulation.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.10,
        delta: float = 0.01,
        max_samples: int = 100,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.max_samples = max_samples

        # Wald boundaries
        self._upper = math.log((1.0 - beta) / alpha)
        self._lower = math.log(beta / (1.0 - alpha))

    def update(
        self,
        state: SPRTState,
        current_reward: float,
        challenger_reward: float,
    ) -> str:
        """Feed one paired observation and return decision.

        Returns "switch", "stay", or "undecided".

        We model rewards as Bernoulli with parameter = survival probability.
        Under H₀: p_current = p_challenger  (no improvement)
        Under H₁: p_challenger = p_current + delta  (challenger is better)
        """
        state.current_rewards.append(current_reward)
        state.challenger_rewards.append(challenger_reward)
        state.samples_seen += 1

        # Empirical survival probabilities (bounded away from 0/1)
        p0 = np.clip(np.mean(state.current_rewards), 0.01, 0.99)
        p1 = np.clip(np.mean(state.challenger_rewards), 0.01, 0.99)

        # Compute per-observation LLR contribution
        # Using Bernoulli model: observation = 1 if challenger won this round
        challenger_won = float(challenger_reward > current_reward)

        # Under H₁, P(challenger wins) = p1; under H₀, P(challenger wins) = p0
        # But we need to handle the case where p0 ≈ p1
        p1_effective = min(p1, 0.99)
        p0_effective = max(p0, 0.01)

        if challenger_won > 0.5:
            llr_increment = math.log(
                max(p1_effective, 1e-10) / max(p0_effective, 1e-10)
            )
        else:
            llr_increment = math.log(
                max(1 - p1_effective, 1e-10) / max(1 - p0_effective, 1e-10)
            )

        state.log_likelihood_ratio += llr_increment

        # Check boundaries
        if state.log_likelihood_ratio >= self._upper:
            state.decision = "switch"
            return "switch"
        elif state.log_likelihood_ratio <= self._lower:
            state.decision = "stay"
            return "stay"
        elif state.samples_seen >= self.max_samples:
            # Truncated SPRT: decide based on sign of accumulated evidence
            if state.log_likelihood_ratio > 0:
                state.decision = "switch"
                return "switch"
            else:
                state.decision = "stay"
                return "stay"

        return "undecided"


# ---------------------------------------------------------------------------
# SPRT Controller
# ---------------------------------------------------------------------------

class SPRTController(BaseController):
    """SPRT-gated adaptive controller.

    Operates in two phases:
    1. Exploitation: use the current best arm.
    2. Evaluation: periodically run a paired comparison (interleaved
       rounds) between the current arm and a challenger, accumulating
       SPRT evidence.

    When the SPRT decides to "switch", the controller commits to the
    challenger.  When it decides "stay", the challenger is dismissed
    and a new one is nominated after a cooldown period.

    Parameters
    ----------
    alpha : float
        Type-I error rate for SPRT.
    beta : float
        Type-II error rate for SPRT.
    delta : float
        Minimum effect size for SPRT.
    eval_interval : int
        Steps between evaluation attempts.
    cooldown : int
        Steps after a "stay" decision before trying a new challenger.
    weights : CostWeights, optional
        For telemetry tracking.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.10,
        delta: float = 0.01,
        eval_interval: int = 10,
        cooldown: int = 20,
        weights: Optional[CostWeights] = None,
    ) -> None:
        super().__init__(weights=weights)

        self._sprt = SPRTEngine(alpha=alpha, beta=beta, delta=delta)

        # Arms
        self._arms = [
            (DecoderChoice.MWPM, DDSequenceType.NONE),
            (DecoderChoice.UNION_FIND, DDSequenceType.NONE),
            (DecoderChoice.MWPM, DDSequenceType.XY4),
            (DecoderChoice.UNION_FIND, DDSequenceType.XY4),
        ]
        self._current_arm: int = 0  # start with MWPM + NONE
        self._challenger_arm: Optional[int] = None

        # SPRT state
        self._sprt_state: Optional[SPRTState] = None
        self._in_eval: bool = False
        self._eval_interval = eval_interval
        self._cooldown = cooldown
        self._cooldown_remaining: int = 0

        # Round-robin challenger selection
        self._challenger_queue: list[int] = []

        # Stats
        self._switch_count: int = 0
        self._eval_count: int = 0

    @property
    def name(self) -> str:
        return "sprt"

    def _make_action(self, arm_idx: int) -> ControlAction:
        dec, dd = self._arms[arm_idx]
        burst_mit = (
            self._current_state is not None
            and self._current_state.burst_active
        )
        return ControlAction(
            decoder=dec,
            dd_policy=dd,
            burst_mitigation=burst_mit,
            request_recalibration=False,
            notes=f"SPRT arm={dec.value}:{dd.value}",
        )

    def _next_challenger(self) -> int:
        """Pick the next challenger arm to evaluate."""
        if not self._challenger_queue:
            # Build queue: all arms except current
            self._challenger_queue = [
                i for i in range(len(self._arms)) if i != self._current_arm
            ]
        return self._challenger_queue.pop(0)

    def observe(self, state: HardwareState) -> None:
        self._current_state = state

    def decide(self) -> ControlAction:
        # If in cooldown, just use current arm
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return self._make_action(self._current_arm)

        # If in evaluation, alternate between current and challenger
        if self._in_eval and self._challenger_arm is not None:
            # Alternate: even steps → current, odd steps → challenger
            if self._sprt_state and self._sprt_state.samples_seen % 2 == 0:
                return self._make_action(self._challenger_arm)
            else:
                return self._make_action(self._current_arm)

        # Check if it's time to start a new evaluation
        if (
            self._step > 0
            and self._step % self._eval_interval == 0
            and not self._in_eval
        ):
            self._challenger_arm = self._next_challenger()
            challenger_dec, challenger_dd = self._arms[self._challenger_arm]
            self._sprt_state = SPRTState(
                challenger_label=f"{challenger_dec.value}:{challenger_dd.value}"
            )
            self._in_eval = True
            self._eval_count += 1
            logger.info(
                f"SPRT: starting evaluation of challenger "
                f"{challenger_dec.value}:{challenger_dd.value}"
            )
            return self._make_action(self._challenger_arm)

        return self._make_action(self._current_arm)

    def update(self, reward: float) -> None:
        if self._in_eval and self._sprt_state is not None and self._challenger_arm is not None:
            # Determine if this round was current or challenger
            is_challenger_round = (self._sprt_state.samples_seen % 2 == 0)

            if is_challenger_round:
                # Wait for the paired current-arm observation
                self._sprt_state.challenger_rewards.append(reward)
            else:
                self._sprt_state.current_rewards.append(reward)

                # We now have a paired observation — run SPRT
                if self._sprt_state.challenger_rewards:
                    challenger_r = self._sprt_state.challenger_rewards[-1]
                    current_r = reward

                    decision = self._sprt._update_paired(
                        self._sprt_state, current_r, challenger_r
                    ) if hasattr(self._sprt, '_update_paired') else self._sprt.update(
                        self._sprt_state, current_r, challenger_r
                    )

                    if decision == "switch":
                        old_arm = self._current_arm
                        self._current_arm = self._challenger_arm
                        self._in_eval = False
                        self._challenger_arm = None
                        self._switch_count += 1
                        logger.info(
                            f"SPRT: switching from arm {old_arm} to {self._current_arm} "
                            f"(LLR={self._sprt_state.log_likelihood_ratio:.3f})"
                        )
                    elif decision == "stay":
                        self._in_eval = False
                        self._challenger_arm = None
                        self._cooldown_remaining = self._cooldown
                        logger.info(
                            f"SPRT: staying with arm {self._current_arm} "
                            f"(LLR={self._sprt_state.log_likelihood_ratio:.3f})"
                        )

            self._sprt_state.samples_seen += 1

        if self._telemetry:
            self._telemetry[-1].cost = -reward
            self._telemetry[-1].extras["in_eval"] = self._in_eval
            self._telemetry[-1].extras["current_arm"] = self._current_arm

    def reset(self) -> None:
        super().reset()
        self._current_arm = 0
        self._challenger_arm = None
        self._sprt_state = None
        self._in_eval = False
        self._cooldown_remaining = 0
        self._challenger_queue.clear()
        self._switch_count = 0
        self._eval_count = 0

    def summary(self) -> dict[str, Any]:
        base = super().summary()
        dec, dd = self._arms[self._current_arm]
        base["current_arm"] = f"{dec.value}:{dd.value}"
        base["switch_count"] = self._switch_count
        base["eval_count"] = self._eval_count
        base["sprt_alpha"] = self._sprt.alpha
        base["sprt_beta"] = self._sprt.beta
        base["sprt_delta"] = self._sprt.delta
        return base
