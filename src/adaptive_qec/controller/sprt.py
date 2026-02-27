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
    challenger_label: str = "challenger"
    log_likelihood_ratio: float = 0.0
    samples_seen: int = 0
    observations: int = 0
    current_rewards: list[float] = field(default_factory=list)
    challenger_rewards: list[float] = field(default_factory=list)
    decision: str = "undecided"  # "undecided" | "switch" | "stay"

    def reset(self) -> None:
        self.log_likelihood_ratio = 0.0
        self.samples_seen = 0
        self.observations = 0
        self.current_rewards.clear()
        self.challenger_rewards.clear()
        self.decision = "undecided"


class SPRTEngine:
    """Wald's Sequential Probability Ratio Test engine.

    Parameters
    ----------
    alpha : float
        Type-I error probability (false switch). Default 0.05.
    beta : float
        Type-II error probability (missed switch). Default 0.10.
    delta : float
        Minimum detectable effect size (in error rate difference).
        Switching is only justified if the challenger's error rate
        is at least delta lower. Default 0.01 (1% absolute).
    max_samples : int
        Maximum samples before forcing a decision (truncated SPRT).
        Prevents indefinite accumulation.
    p0 : float, optional
        Baseline Bernoulli null rate.
    p1 : float, optional
        Challenger Bernoulli alternative rate.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.10,
        delta: float = 0.01,
        max_samples: int = 100,
        p0: Optional[float] = None,
        p1: Optional[float] = None,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.max_samples = max_samples
        self.p0 = p0
        self.p1 = p1

        # Wald boundaries
        self._upper = math.log((1.0 - beta) / alpha)
        self._lower = math.log(beta / (1.0 - alpha))

    def update(
        self,
        state: SPRTState,
        current_reward: Optional[float] = None,
        challenger_reward: Optional[float] = None,
        success: Optional[bool] = None,
    ) -> str:
        """Feed observation and return decision ("continue", "switch", or "stay")."""
        if success is not None:
            state.observations += 1
            state.samples_seen = state.observations
            p0 = self.p0 if self.p0 is not None else 0.03
            p1 = self.p1 if self.p1 is not None else (p0 + self.delta)
            if success:
                llr_incr = math.log(p1 / p0)
            else:
                llr_incr = math.log((1.0 - p1) / (1.0 - p0))
            state.log_likelihood_ratio += llr_incr

            if state.log_likelihood_ratio >= self._upper:
                state.decision = "switch"
                return "switch"
            elif state.log_likelihood_ratio <= self._lower:
                state.decision = "stay"
                return "stay"
            elif state.observations >= self.max_samples:
                state.decision = "switch" if state.log_likelihood_ratio > 0 else "stay"
                return state.decision
            return "continue"

        if current_reward is None or challenger_reward is None:
            return "continue"

        state.current_rewards.append(current_reward)
        state.challenger_rewards.append(challenger_reward)
        state.samples_seen += 1
        state.observations = state.samples_seen

        # Empirical survival probabilities (bounded away from 0/1)
        p0 = np.clip(np.mean(state.current_rewards), 0.01, 0.99)
        p1 = np.clip(np.mean(state.challenger_rewards), 0.01, 0.99)

        challenger_won = float(challenger_reward > current_reward)
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
            if state.log_likelihood_ratio > 0:
                state.decision = "switch"
                return "switch"
            else:
                state.decision = "stay"
                return "stay"

        return "continue"


# ---------------------------------------------------------------------------
# SPRT Controller
# ---------------------------------------------------------------------------

class SPRTController(BaseController):
