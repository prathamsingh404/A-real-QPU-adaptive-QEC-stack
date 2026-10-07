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

def test_bandit_cumulative_regret_tracking():
    bandit = DriftAdaptiveBandit(num_arms=2)
    bandit.update(0, reward=0.8, oracle_best_reward=1.0)
    assert bandit.cumulative_regret == pytest.approx(0.2)
    bandit.update(1, reward=1.0, oracle_best_reward=1.0)
    assert bandit.cumulative_regret == pytest.approx(0.2)

def test_bandit_arm_means_calculation():
    bandit = DriftAdaptiveBandit(num_arms=2)
    bandit.update(0, 0.5)
    bandit.update(0, 0.7)
    means = bandit.get_arm_means()
    assert means[0] == pytest.approx(0.6)
    assert means[1] == 0.0

def test_bandit_reset_regret():
    bandit = DriftAdaptiveBandit(num_arms=2)
    bandit.update(0, 0.5, oracle_best_reward=1.0)
    bandit.reset_regret()
    assert bandit.cumulative_regret == 0.0

def test_bandit_round_robin_exploration():
    bandit = DriftAdaptiveBandit(num_arms=3)
    assert bandit.select_arm() == 0
    bandit.update(0, 1.0)
    assert bandit.select_arm() == 1
    bandit.update(1, 1.0)
    assert bandit.select_arm() == 2

def test_bandit_empty_active_recovery():
    bandit = DriftAdaptiveBandit(num_arms=2)
    bandit.active_arms[:] = False
    arm = bandit.select_arm()
    assert arm in [0, 1]
    assert np.all(bandit.active_arms)

def test_bandit_window_trimming():
    bandit = DriftAdaptiveBandit(num_arms=1, window_size=5)
    for i in range(20):
        bandit.update(0, float(i))
    assert len(bandit.arm_rewards[0]) <= 10

def test_bandit_set_arm_window():
    bandit = DriftAdaptiveBandit(num_arms=2)
    bandit.set_arm_window(1, [0.5, 0.8])
    assert len(bandit.arm_rewards[1]) == 2

def test_bandit_single_arm():
    bandit = DriftAdaptiveBandit(num_arms=1)
    assert bandit.select_arm() == 0
    bandit.update(0, 1.0)
    assert bandit.select_arm() == 0
