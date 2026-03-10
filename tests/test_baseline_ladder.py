"""
Tests for Baseline Ladder and Counterfactual Engine (tests/test_baseline_ladder.py).
"""

import numpy as np
import pytest
import stim

from adaptive_qec.evaluation.baseline_ladder import BaselineLadder, BaselineRung
from adaptive_qec.evaluation.counterfactual import CounterfactualEngine, wilson_score_interval


def test_wilson_score_interval_properties():
    """Verify that Wilson score interval is strictly in [0, 1] and valid."""
    low, high = wilson_score_interval(10, 100)
    assert 0.0 <= low <= 0.1 <= high <= 1.0

    low_zero, high_zero = wilson_score_interval(0, 50)
    assert low_zero == 0.0
    assert high_zero > 0.0


def test_baseline_ladder_instantiation_and_counterfactual_eval():
    """Verify that BaselineLadder sets up B0-B3 and counterfactually evaluates syndromes."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.007,
    )
    ladder = BaselineLadder(circuit=circuit)
