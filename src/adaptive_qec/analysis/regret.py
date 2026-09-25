"""
Regret analysis for adaptive QEC controller evaluation.

Computes formal regret metrics to quantify how much worse each
controller performs compared to the oracle (best fixed arm in
hindsight) and the best dynamic policy.

Metrics:
    - Cumulative regret: Σ_t [r*(t) - r_controller(t)]
    - Time-averaged regret: cumulative / T
    - Normalized regret: regret / oracle_reward
    - Per-phase regret: regret broken down by noise scenario phase
    - Switching cost: total mode switches × penalty

References:
    - Auer et al., "The Nonstochastic Multiarmed Bandit Problem" (2002)
    - Roadmap: PROJECT_EVOLUTION_ROADMAP.md §Phase-5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegretAnalysis:
    """Complete regret analysis for a controller run.

    All values are computed from actual experiment data,
    not estimated or simulated.
    """
    controller_name: str
    total_steps: int

    # Cumulative regret vs best fixed arm in hindsight
    cumulative_regret: float
    time_averaged_regret: float
    normalized_regret: float

    # Per-step regret curve
    regret_curve: list[float]  # cumulative regret at each step

    # Oracle performance
    oracle_arm: str
    oracle_total_reward: float

    # Controller performance
    controller_total_reward: float
    controller_mean_reward: float

    # Switching analysis
    total_switches: int
    switch_regret: float  # switches × penalty

    # Statistical
    reward_std: float
    reward_p5: float
    reward_p95: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "controller_name": self.controller_name,
            "total_steps": self.total_steps,
            "cumulative_regret": self.cumulative_regret,
            "time_averaged_regret": self.time_averaged_regret,
            "normalized_regret": self.normalized_regret,
            "oracle_arm": self.oracle_arm,
            "oracle_total_reward": self.oracle_total_reward,
            "controller_total_reward": self.controller_total_reward,
            "controller_mean_reward": self.controller_mean_reward,
            "total_switches": self.total_switches,
            "switch_regret": self.switch_regret,
            "reward_std": self.reward_std,
            "reward_p5": self.reward_p5,
            "reward_p95": self.reward_p95,
        }


class RegretAnalyzer:
    """Computes regret metrics from experiment results.

    Usage:
        analyzer = RegretAnalyzer(window_results)
        analysis = analyzer.analyze("exp3")
    """

    def __init__(
        self,
        window_results: list[dict[str, Any]],
        switch_penalty: float = 0.001,
    ) -> None:
        self._results = window_results
        self._switch_penalty = switch_penalty

        # Group results by controller
        self._by_controller: dict[str, list[dict]] = {}
        for r in window_results:
            name = r["controller_name"]
            self._by_controller.setdefault(name, []).append(r)

    def _compute_oracle(self, arm_rewards: dict[str, list[float]]) -> tuple[str, float]:
        """Find the best fixed arm in hindsight.

        The oracle always plays the single arm with the highest
        total reward over the entire experiment.
        """
        best_arm = ""
        best_total = -float("inf")

        for arm, rewards in arm_rewards.items():
            total = sum(rewards)
            if total > best_total:
                best_total = total
                best_arm = arm

        return best_arm, best_total

    def _extract_arm_rewards(
        self, results: list[dict], all_results: list[dict]
    ) -> dict[str, list[float]]:
        """Build per-arm reward histories from ALL controllers' results.

        For the oracle comparison, we need the reward that each arm
        *would have* achieved at each step.  In an interleaved experiment,
        different controllers pull different arms at the same step.
        """
        arm_rewards: dict[str, list[float]] = {}

        for r in all_results:
            arm_label = f"{r['action']['decoder']}:{r['action']['dd_policy']}"
            arm_rewards.setdefault(arm_label, []).append(r["reward"])

        return arm_rewards

    def analyze(self, controller_name: str) -> RegretAnalysis:
        """Compute regret analysis for a specific controller.

        Parameters
        ----------
        controller_name : str
            Name of the controller to analyze.

        Returns
        -------
        RegretAnalysis
            Comprehensive regret metrics.
        """
        if controller_name not in self._by_controller:
            raise ValueError(f"No results for controller: {controller_name}")

        results = self._by_controller[controller_name]
        rewards = np.array([r["reward"] for r in results])
        T = len(rewards)

        # Compute oracle (best fixed arm in hindsight)
        arm_rewards = self._extract_arm_rewards(results, self._results)
        oracle_arm, oracle_total = self._compute_oracle(arm_rewards)

        # Per-step oracle reward (best arm's mean reward)
        oracle_per_step = oracle_total / max(T, 1)

        # Cumulative regret: Σ_t [oracle_per_step - reward_t]
        regret_curve: list[float] = []
        cumulative = 0.0
        for r in rewards:
            cumulative += oracle_per_step - r
            regret_curve.append(cumulative)

        cumulative_regret = cumulative
        time_averaged = cumulative_regret / max(T, 1)

        controller_total = float(np.sum(rewards))
