import pytest
import numpy as np
from adaptive_qec.controller.bandit import DriftAdaptiveBandit

def test_bandit_initial_state():
    bandit = DriftAdaptiveBandit(num_arms=4)
    assert bandit.num_arms == 4
    assert bandit.get_active_arm_count() == 4
    assert bandit.cumulative_regret == 0.0
