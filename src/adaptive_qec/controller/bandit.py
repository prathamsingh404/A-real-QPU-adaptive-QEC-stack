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

def select_arm(self) -> int:
        active_indices = np.where(self.active_arms)[0]
        if len(active_indices) == 0:
            self.active_arms[:] = True
            active_indices = np.where(self.active_arms)[0]
        for arm in active_indices:
            if len(self.arm_rewards[arm]) == 0:
                self.last_selected_arm = arm
                return arm
        means = np.array([np.mean(self.arm_rewards[arm][-self.window_size:]) for arm in active_indices])
        best_idx = np.argmax(means)
        chosen_arm = int(active_indices[best_idx])
        self.last_selected_arm = chosen_arm
        return chosen_arm

def update(self, arm: int, reward: float, oracle_best_reward: Optional[float] = None) -> None:
        self.arm_rewards[arm].append(float(reward))
        self.pull_counts[arm] += 1
        if len(self.arm_rewards[arm]) > self.window_size * 2:
            self.arm_rewards[arm] = self.arm_rewards[arm][-self.window_size:]
        if oracle_best_reward is not None:
            regret = max(0.0, oracle_best_reward - reward)
            self.cumulative_regret += regret

def on_drift_detected(self) -> None:
        self.drift_count += 1
        eliminated = ~self.active_arms
        self.active_arms[eliminated] = True
        for arm in range(self.num_arms):
            if eliminated[arm]:
                self.arm_rewards[arm] = []
        logger.info('DASE drift detected: reactivated eliminated arms while preserving winner history.')

def get_arm_means(self) -> np.ndarray:
        return np.array([np.mean(r[-self.window_size:]) if len(r) > 0 else 0.0 for r in self.arm_rewards])
