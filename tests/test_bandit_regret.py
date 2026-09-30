import pytest
import numpy as np
from adaptive_qec.controller.bandit import DriftAdaptiveBandit

def test_bandit_initial_state():
    bandit = DriftAdaptiveBandit(num_arms=4)
    assert bandit.num_arms == 4
    assert bandit.get_active_arm_count() == 4
    assert bandit.cumulative_regret == 0.0

def test_bandit_arm_retention_on_drift():
    bandit = DriftAdaptiveBandit(num_arms=3, window_size=20)
    for _ in range(10):
        bandit.update(0, 1.0)
        bandit.update(1, 0.0)
        bandit.update(2, 0.0)
    bandit.active_arms[1] = False
    bandit.active_arms[2] = False
    bandit.on_drift_detected()
    assert np.all(bandit.active_arms)
    assert len(bandit.arm_rewards[0]) == 10
    assert len(bandit.arm_rewards[1]) == 0
    assert len(bandit.arm_rewards[2]) == 0
