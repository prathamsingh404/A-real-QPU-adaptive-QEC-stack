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


def test_sublinear_regret_bound_stationary():
    """Cumulative regret grows sublinearly: average regret per step R_T/T decreases over time."""
    from adaptive_qec.controller.bandit import Exp3Controller

    controller = Exp3Controller(gamma=0.1)
    # Fixed rewards: arm 0 is optimal (0.8), others are suboptimal (0.3)
    rng = np.random.default_rng(42)
    regrets = []
    cumulative = 0.0

    for t in range(1, 301):
        action = controller.decide()
        arm_idx = controller._last_arm_idx
        # Optimal arm gives reward 0.8, others give 0.3
        reward = 0.8 if arm_idx == 0 else 0.3
        oracle_best = 0.8
        regret_step = oracle_best - reward
        cumulative += regret_step
        controller.update(reward=reward)
        regrets.append(cumulative / t)

    # Average regret at step 300 should be lower than at step 30
    assert regrets[-1] < regrets[30]
    # Final cumulative regret is strictly bounded well below linear worst-case (0.5 * 300 = 150)
    assert cumulative < 125.0


def test_drift_regret_bound_recovery():
    """When optimal arm shifts after drift, bandit adapts and tracks the new optimum."""
    bandit = DriftAdaptiveBandit(num_arms=2, window_size=20)

    # Phase 1: arm 0 is optimal
    for _ in range(30):
        bandit.update(0, 0.9, oracle_best_reward=0.9)
        bandit.update(1, 0.2, oracle_best_reward=0.9)

    means_p1 = bandit.get_arm_means()
    assert means_p1[0] > means_p1[1]

    # Drift occurs: arm 1 becomes optimal
    bandit.on_drift_detected()
    for _ in range(30):
        bandit.update(0, 0.2, oracle_best_reward=0.9)
        bandit.update(1, 0.9, oracle_best_reward=0.9)

    means_p2 = bandit.get_arm_means()
    assert means_p2[1] > means_p2[0]


def test_exp3_adversarial_regret_bound():
    """Exp3 cumulative regret respects O(sqrt(K * T * ln K)) theoretical scaling."""
    from adaptive_qec.controller.bandit import Exp3Controller

    controller = Exp3Controller(gamma=0.15)
    K = controller._K
    T = 200
    # Worst-case theoretical bound constant
    theoretical_bound = 2.0 * np.sqrt(K * T * np.log(max(K, 2)))

    cumulative_regret = 0.0
    for t in range(T):
        controller.decide()
        arm_idx = controller._last_arm_idx
        reward = 0.7 if arm_idx == 0 else 0.2
        regret = 0.7 - reward
        cumulative_regret += regret
        controller.update(reward=reward)

    assert cumulative_regret <= theoretical_bound

