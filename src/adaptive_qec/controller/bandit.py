"""
Non-stationary Bandit Controller with Amnesia-Free Arm Retention
"""
from __future__ import annotations
import logging
from typing import Optional, List
import numpy as np

logger = logging.getLogger(__name__)

class DriftAdaptiveBandit:
    """DA-SE Bandit with continuous active-arm window retention upon drift events."""
    def __init__(self, num_arms: int, window_size: int = 50, confidence_param: float = 0.5) -> None:
        self.num_arms = num_arms
        self.window_size = window_size
        self.confidence_param = confidence_param
        self.active_arms = np.ones(num_arms, dtype=bool)
        self.arm_rewards: List[List[float]] = [[] for _ in range(num_arms)]
        self.pull_counts = np.zeros(num_arms, dtype=int)
        self.drift_count = 0
        self.cumulative_regret = 0.0
        self.last_selected_arm = 0
