import pytest
import numpy as np

def test_cumulative_regret_monotonicity():
    mwpm_errors = np.array([10, 15, 20, 25])
    uf_errors = np.array([25, 20, 15, 10])
    adaptive_errors = np.array([10, 15, 15, 10])
    oracle_best = np.minimum(mwpm_errors, uf_errors)
    regret_per_window = np.maximum(0, adaptive_errors - oracle_best)
    cum_regret = np.cumsum(regret_per_window)
    assert np.all(cum_regret >= 0)
    assert np.all(np.diff(cum_regret) >= 0)
    assert cum_regret[-1] == 0
