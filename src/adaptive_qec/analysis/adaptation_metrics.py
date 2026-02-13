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
