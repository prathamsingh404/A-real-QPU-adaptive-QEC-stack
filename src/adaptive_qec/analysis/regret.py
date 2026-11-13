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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

    # Dynamic & theoretical bounds
    dynamic_cumulative_regret: float = 0.0
    theoretical_bound: float = 0.0

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
            "dynamic_cumulative_regret": self.dynamic_cumulative_regret,
            "theoretical_bound": self.theoretical_bound,
        }


class RegretAnalyzer:
    """Computes regret metrics from experiment results.

    Usage:
        analyzer = RegretAnalyzer(window_results)
        analysis = analyzer.analyze("exp3")
    """

    def __init__(
        self,
        window_results: Optional[list[dict[str, Any]]] = None,
        switch_penalty: float = 0.001,
        num_arms: Optional[int] = None,
    ) -> None:
        self._results = list(window_results) if window_results is not None else []
        self._switch_penalty = switch_penalty
        self.num_arms = num_arms

        # Group results by controller
        self._by_controller: dict[str, list[dict]] = {}
        for r in self._results:
            name = r.get("controller_name", "unknown")
            self._by_controller.setdefault(name, []).append(r)

    def add_result(self, result: dict[str, Any]) -> None:
        """Add a single window result record."""
        self._results.append(result)
        name = result.get("controller_name", "unknown")
        self._by_controller.setdefault(name, []).append(result)

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
        normalized = cumulative_regret / max(oracle_total, 1e-10)

        # Count switches
        switches = 0
        for i in range(1, len(results)):
            prev_arm = f"{results[i-1]['action']['decoder']}:{results[i-1]['action']['dd_policy']}"
            curr_arm = f"{results[i]['action']['decoder']}:{results[i]['action']['dd_policy']}"
            if prev_arm != curr_arm:
                switches += 1

        switch_regret = switches * self._switch_penalty

        # Theoretical regret bound (piecewise-stationary MAB bound)
        num_k = self.num_arms if self.num_arms is not None else max(len(arm_rewards), 1)
        bounds = compute_cumulative_regret_bounds(
            total_steps=T,
            num_arms=num_k,
            num_change_points=max(1, switches),
            switch_cost=self._switch_penalty,
        )
        theoretical_bound = bounds["piecewise_stationary_bound"]

        # Dynamic oracle: per-step best arm across available arms
        dynamic_cum_regret = 0.0
        if arm_rewards:
            min_len = min(len(v) for v in arm_rewards.values())
            if min_len >= T:
                step_oracles = [max(arm_rewards[a][t] for a in arm_rewards) for t in range(T)]
                dynamic_cum_regret = float(np.sum(np.maximum(0.0, np.array(step_oracles) - rewards))) + switch_regret
            else:
                dynamic_cum_regret = cumulative_regret
        else:
            dynamic_cum_regret = cumulative_regret

        return RegretAnalysis(
            controller_name=controller_name,
            total_steps=T,
            cumulative_regret=cumulative_regret,
            time_averaged_regret=time_averaged,
            normalized_regret=normalized,
            regret_curve=regret_curve,
            oracle_arm=oracle_arm,
            oracle_total_reward=oracle_total,
            controller_total_reward=controller_total,
            controller_mean_reward=float(np.mean(rewards)),
            total_switches=switches,
            switch_regret=switch_regret,
            reward_std=float(np.std(rewards)),
            reward_p5=float(np.percentile(rewards, 5)),
            reward_p95=float(np.percentile(rewards, 95)),
            dynamic_cumulative_regret=dynamic_cum_regret,
            theoretical_bound=theoretical_bound,
        )

    def compare(self) -> dict[str, RegretAnalysis]:
        """Analyze all controllers and return comparative results."""
        analyses = {}
        for name in self._by_controller:
            analyses[name] = self.analyze(name)
        return analyses

    def summary_table(self) -> str:
        """Generate a human-readable comparison table."""
        analyses = self.compare()

        lines = [
            "=" * 80,
            f"{'Controller':<25} {'Avg Reward':>10} {'Cum. Regret':>12} "
            f"{'Norm. Regret':>13} {'Switches':>9}",
            "-" * 80,
        ]

        for name, a in sorted(analyses.items(), key=lambda x: -x[1].controller_mean_reward):
            lines.append(
                f"{name:<25} {a.controller_mean_reward:>10.6f} "
                f"{a.cumulative_regret:>12.4f} "
                f"{a.normalized_regret:>13.4f} "
                f"{a.total_switches:>9}"
            )

        lines.append("=" * 80)
        lines.append(f"Oracle arm: {list(analyses.values())[0].oracle_arm if analyses else 'N/A'}")

        return "\n".join(lines)


def compute_cumulative_regret_bounds(
    total_steps: int,
    num_arms: int = 5,
    num_change_points: int = 1,
    switch_cost: float = 0.001,
) -> Dict[str, float]:
    """
    Compute theoretical cumulative regret bounds for multi-armed bandit controllers.

    Includes:
        - Exp3 minimax adversarial bound: 2 * sqrt(K * T * log(K))
        - Piecewise-stationary switching bound:
          sqrt(S * K * T * log(max(2, T))) + S * switch_cost
        - Sublinear rate: sqrt(K * T)

    Args:
        total_steps: Total horizon T.
        num_arms: Number of arms K.
        num_change_points: Number of non-stationary environment switches S.
        switch_cost: Fixed penalty per switch C_switch.

    Returns:
        Dict[str, float]: Theoretical upper bounds on cumulative regret.
    """
    if total_steps <= 0:
        return {
            "exp3_bound": 0.0,
            "piecewise_stationary_bound": 0.0,
            "sublinear_bound": 0.0,
        }

    K = max(1, num_arms)
    T = total_steps
    S = max(1, num_change_points)

    exp3_bound = float(2.0 * np.sqrt(K * T * np.log(max(2, K))))
    piecewise_bound = float(
        np.sqrt(S * K * T * np.log(max(2, T))) + S * switch_cost
    )
    sublinear_bound = float(np.sqrt(K * T))

    return {
        "exp3_bound": exp3_bound,
        "piecewise_stationary_bound": piecewise_bound,
        "sublinear_bound": sublinear_bound,
    }


def compute_dynamic_regret(
    controller_losses: Sequence[float],
    oracle_losses: Sequence[float],
    switch_cost: float = 0.0,
    switches: int = 0,
) -> Tuple[float, List[float]]:
    """
    Compute empirical dynamic regret against per-step optimal oracle.

    R_T = sum_{t=1}^T [ loss_{controller}(t) - loss_{oracle}(t) ] + switches * switch_cost

    Args:
        controller_losses: Sequence of losses/errors incurred by controller.
        oracle_losses: Sequence of best counterfactual losses at each step.
        switch_cost: Penalty incurred per mode switch.
        switches: Total count of controller switches.

    Returns:
        (cumulative_regret, regret_curve)
    """
    c = np.asarray(controller_losses, dtype=np.float64)
    o = np.asarray(oracle_losses, dtype=np.float64)
    if len(c) != len(o):
        raise ValueError(f"Length mismatch: {len(c)} vs {len(o)}")

    inst_regret = np.maximum(0.0, c - o)
    curve = list(np.cumsum(inst_regret))
    total = float(curve[-1] if len(curve) > 0 else 0.0) + switches * switch_cost
    return (total, curve)
