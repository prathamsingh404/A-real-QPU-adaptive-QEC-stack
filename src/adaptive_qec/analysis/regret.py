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

