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

    assert BaselineRung.B0_FACTORY_MWPM in ladder.arms
    assert BaselineRung.B1_OPTIMIZED_MWPM in ladder.arms
    assert BaselineRung.B2_TRUE_UNION_FIND in ladder.arms
    assert BaselineRung.B3_ONLINE_REWEIGHTED in ladder.arms

    engine = ladder.to_counterfactual_engine()
    sampler = circuit.compile_detector_sampler(seed=123)
    syndromes, observables = sampler.sample(shots=60, separate_observables=True)

    results, disagreement = engine.evaluate(syndromes, observables)

    # Check that all arms are present
    assert BaselineRung.B0_FACTORY_MWPM.value in results
    assert BaselineRung.B1_OPTIMIZED_MWPM.value in results
    assert BaselineRung.B2_TRUE_UNION_FIND.value in results
    assert BaselineRung.B3_ONLINE_REWEIGHTED.value in results

    # Verify B0 reference delta is identically zero
    b0_res = results[BaselineRung.B0_FACTORY_MWPM.value]
    assert b0_res.mean_delta_ler == 0.0

    # Disagreement matrix symmetry and zero diagonal
    b0_name = BaselineRung.B0_FACTORY_MWPM.value
    b2_name = BaselineRung.B2_TRUE_UNION_FIND.value
    assert disagreement.get(b0_name, b0_name) == 0.0
    assert 0.0 <= disagreement.get(b0_name, b2_name) <= 1.0
    assert disagreement.get(b0_name, b2_name) == disagreement.get(b2_name, b0_name)
