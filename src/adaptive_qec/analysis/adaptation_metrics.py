"""
Theoretical Modeling, Analytical Proofs, and Adaptation Value Metrics.

Tasks covered:
- TASK-084 [Point 27]: Gross adaptation value (V_gross) and net adaptation value (V_net).
- TASK-085 [Point 43]: Adaptation efficiency eta_adapt = G_observable / G_oracle.
- TASK-086 [Point 44]: Empirical cumulative regret R_T against hindsight optimum with switching costs.
- TASK-087 [Point 57]: Distribution-free conformal prediction intervals for policy LER estimates.
- TASK-088 [Point 96]: Formalized switching condition: LCB(Delta L_t) > C_switch + C_latency + C_uncertainty.
- TASK-089 [Point 97]: Analytical upper bound on false-switching probability.
- TASK-090 [Point 98]: Analytical adaptation boundary threshold A_drift^* as a function of noise variance and cost.
- TASK-091 [Point 99]: Piecewise-stationary regret bound O(S sqrt(T) + S * C_switch).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdaptationValueReport:
    """Gross and Net adaptation value breakdown."""
    static_ler: float
    raw_adapted_ler: float           # Counterfactual performance of new decoder ignoring taxes
    actual_adapted_ler: float        # Observed performance including switching/latency taxes
    gross_value: float               # V_gross = static_ler - raw_adapted_ler
    net_value: float                 # V_net = static_ler - actual_adapted_ler
    switching_cost: float
    latency_cost: float
    uncertainty_cost: float
    physical_cost: float
    total_tax: float                 # Sum of all adaptation taxes
    adaptation_profitable: bool      # True if V_net > 0


def compute_adaptation_values(
    static_ler: float,
    raw_adapted_ler: float,
    switching_cost: float = 0.001,
    latency_cost: float = 0.0005,
    uncertainty_cost: float = 0.0,
    physical_cost: float = 0.0,
) -> AdaptationValueReport:
    """
    Compute Gross and Net Adaptation Value.

    V_gross = LER_static - LER_raw_candidate
    V_net = V_gross - Total_Tax
    """
    total_tax = switching_cost + latency_cost + uncertainty_cost + physical_cost
    gross_value = static_ler - raw_adapted_ler
    actual_adapted_ler = raw_adapted_ler + total_tax
    net_value = static_ler - actual_adapted_ler
    profitable = net_value > 0.0

    return AdaptationValueReport(
        static_ler=static_ler,
        raw_adapted_ler=raw_adapted_ler,
        actual_adapted_ler=actual_adapted_ler,
        gross_value=gross_value,
        net_value=net_value,
        switching_cost=switching_cost,
        latency_cost=latency_cost,
        uncertainty_cost=uncertainty_cost,
        physical_cost=physical_cost,
        total_tax=total_tax,
        adaptation_profitable=profitable,
    )


@dataclass(frozen=True)
class AdaptationEfficiencyReport:
    """Quantification of observable policy gain relative to oracle optimum."""
    static_ler: float
    deployable_ler: float
    oracle_ler: float
    gain_deployable: float            # G_observable = LER_static - LER_deployable
    gain_oracle: float                # G_oracle = LER_static - LER_oracle
    efficiency: float                 # eta = G_observable / G_oracle
    is_valid_range: bool              # True if efficiency in [0, 1]


def compute_adaptation_efficiency(
    static_ler: float,
    deployable_ler: float,
    oracle_ler: float,
) -> AdaptationEfficiencyReport:
    """
    Compute adaptation efficiency:
        eta_adapt = (LER_static - LER_deployable) / (LER_static - LER_oracle)
    """
    gain_deployable = static_ler - deployable_ler
    gain_oracle = static_ler - oracle_ler

    if gain_oracle <= 1e-12:
        # No headroom to improve over static baseline
        eff = 1.0 if abs(gain_deployable) <= 1e-12 else 0.0
    else:
        eff = gain_deployable / gain_oracle

    valid = 0.0 <= eff <= 1.0

    return AdaptationEfficiencyReport(
        static_ler=static_ler,
        deployable_ler=deployable_ler,
        oracle_ler=oracle_ler,
        gain_deployable=gain_deployable,
        gain_oracle=gain_oracle,
        efficiency=float(eff),
        is_valid_range=valid,
    )


@dataclass
class RegretTrajectory:
    """Empirical cumulative regret against dynamic hindsight oracle."""
    total_steps: int
    cumulative_regret: float
    time_averaged_regret: float
    regret_curve: List[float]
    total_switches: int
    switching_penalty_total: float
    oracle_actions: List[str]
    controller_actions: List[str]


class EmpiricalRegretTracker:
    """
    Formulates and computes empirical cumulative regret R_T against hindsight optimum:
        R_T = sum_{t=1}^T [ L(a_t, s_t) - min_{a^*} L(a^*, s_t) ] + sum_{t=1}^T C_switch * 1_{a_t != a_{t-1}}
    """

    def __init__(self, switch_cost: float = 0.001) -> None:
        self.switch_cost = switch_cost
        self._step_losses_controller: List[float] = []
        self._step_losses_oracle: List[float] = []
        self._actions_controller: List[str] = []
        self._actions_oracle: List[str] = []
        self._switches: int = 0

    def record_step(
        self,
        chosen_action: str,
        chosen_loss: float,
        oracle_action: str,
        oracle_loss: float,
    ) -> None:
        if self._actions_controller and chosen_action != self._actions_controller[-1]:
            self._switches += 1
            effective_loss = chosen_loss + self.switch_cost
        else:
            effective_loss = chosen_loss

        self._actions_controller.append(chosen_action)
        self._actions_oracle.append(oracle_action)
        self._step_losses_controller.append(effective_loss)
        self._step_losses_oracle.append(oracle_loss)

    def compute_regret(self) -> RegretTrajectory:
        t = len(self._step_losses_controller)
        if t == 0:
            return RegretTrajectory(0, 0.0, 0.0, [], 0, 0.0, [], [])

        regret_curve: List[float] = []
        cum_regret = 0.0
        for loss_c, loss_o in zip(self._step_losses_controller, self._step_losses_oracle):
            inst_regret = max(0.0, loss_c - loss_o)
            cum_regret += inst_regret
            regret_curve.append(cum_regret)

        return RegretTrajectory(
            total_steps=t,
            cumulative_regret=cum_regret,
            time_averaged_regret=cum_regret / t,
            regret_curve=regret_curve,
            total_switches=self._switches,
            switching_penalty_total=self._switches * self.switch_cost,
            oracle_actions=list(self._actions_oracle),
            controller_actions=list(self._actions_controller),
        )


class ConformalLERPredictor:
    """
    Distribution-free split conformal prediction for policy LER estimates.
    Provides finite-sample valid coverage:
        P( LER_{true} in [ LER_hat - q, LER_hat + q ] ) >= 1 - alpha
    without assuming asymptotic normality.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self.q_hat: Optional[float] = None
        self._calibration_scores: List[float] = []

    def calibrate(self, predicted_lers: Sequence[float], true_lers: Sequence[float]) -> float:
        """
        Calibrate on held-out validation trajectories (seeds 10–19).
        Non-conformity score: s_i = |predicted_ler_i - true_ler_i|
        """
        if len(predicted_lers) != len(true_lers) or len(predicted_lers) == 0:
            raise ValueError("Predictions and ground truths must have equal non-zero length")

        scores = [abs(float(p) - float(y)) for p, y in zip(predicted_lers, true_lers)]
        self._calibration_scores = sorted(scores)
        n = len(scores)

        # Conformal quantile level: ceil((n + 1) * (1 - alpha)) / n
        level = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        level = min(level, n)
        self.q_hat = float(self._calibration_scores[level - 1])
        return self.q_hat

    def predict_interval(self, predicted_ler: float) -> Tuple[float, float]:
        """Return distribution-free [lower, upper] prediction interval."""
        if self.q_hat is None:
            # Fallback conservative bound before calibration
            margin = 0.02
        else:
            margin = self.q_hat

        lower = max(0.0, predicted_ler - margin)
        upper = min(1.0, predicted_ler + margin)
        return (float(lower), float(upper))


class AnalyticalProofsAndBounds:
    """
    Analytical theorems and bounds for Adaptive QEC.
    Tasks: TASK-088, TASK-089, TASK-090, TASK-091.
    """

    @staticmethod
    def evaluate_switching_condition(
        delta_hat: float,
        sigma_delta: float,
        c_switch: float,
        c_latency: float,
        c_uncertainty: float = 0.0,
        c_physical: float = 0.0,
        confidence: float = 0.95,
    ) -> Tuple[bool, float, float]:
        """
        TASK-088: Formalize foundational switching condition:
            LCB(Delta L_t) > C_switch + C_latency + C_uncertainty + C_physical
        where LCB(Delta L_t) = Delta_hat - z_{1-alpha} * sigma_delta.

        Returns:
            (should_switch, lcb_value, total_threshold)
        """
        alpha = 1.0 - confidence
        z = float(stats.norm.ppf(1.0 - alpha))  # One-sided z-score
        lcb = delta_hat - z * sigma_delta
        total_threshold = c_switch + c_latency + c_uncertainty + c_physical
        should_switch = lcb > total_threshold
        return (should_switch, lcb, total_threshold)

    @staticmethod
    def false_switching_probability_bound(
        sigma_noise: float,
        n_samples: int,
        threshold: float,
    ) -> float:
        """
        TASK-089: Derive analytical upper bound on false-switching probability.
        Under the null hypothesis (true Delta L = 0):
        By Sub-Gaussian concentration / Chernoff bound:
            P( Delta_hat > threshold | Delta = 0 ) <= exp( - (n * threshold^2) / (2 * sigma_noise^2) )
        """
        if n_samples <= 0 or sigma_noise <= 0.0 or threshold <= 0.0:
            return 1.0
        exponent = -(n_samples * (threshold ** 2)) / (2.0 * (sigma_noise ** 2))
        return float(np.exp(np.clip(exponent, -700.0, 0.0)))

    @staticmethod
    def adaptation_boundary_threshold(
        sigma_noise: float,
        c_switch: float,
        c_latency: float,
        window_length: int,
        desired_confidence: float = 0.95,
    ) -> float:
        """
        TASK-090: Derive analytical adaptation boundary threshold A_drift^*:
        The minimum true drift amplitude required for profitable adaptation:
            A_drift^* = C_switch + C_latency + z_{1-alpha} * (sigma_noise / sqrt(window_length))
        """
        alpha = 1.0 - desired_confidence
        z = float(stats.norm.ppf(1.0 - alpha))
        se = sigma_noise / np.sqrt(window_length) if window_length > 0 else 0.0
        return float(c_switch + c_latency + z * se)

    @staticmethod
    def piecewise_stationary_regret_bound(
        total_steps_t: int,
        num_change_points_s: int,
        num_arms_k: int = 5,
        c_switch: float = 0.001,
    ) -> float:
        """
        TASK-091: Derive regret bound under piecewise-stationary drift with S change points:
            R_T <= O( sqrt(S * K * T * log(T)) + S * C_switch )
        """
        if total_steps_t <= 0:
            return 0.0
        term_estimation = np.sqrt(num_change_points_s * num_arms_k * total_steps_t * np.log(max(2, total_steps_t)))
        term_switching = num_change_points_s * c_switch
        return float(term_estimation + term_switching)
