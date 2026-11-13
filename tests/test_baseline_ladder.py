"""
Tests for Baseline Ladder and Counterfactual Engine (tests/test_baseline_ladder.py).
"""

import numpy as np
import pytest
import stim

from adaptive_qec.decoders.mwpm import MWPMDecoder
from adaptive_qec.evaluation.baseline_ladder import (
    BaselineLadder,
    BaselineRung,
    BaselineLevel,
    BASELINE_LEVEL_REGISTRY,
    BaselineDeltaBenchmark,
    compute_baseline_deltas,
)
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

    # Test new ladder helper methods
    arm = ladder.get_arm(BaselineRung.B0_FACTORY_MWPM)
    assert arm.rung == BaselineRung.B0_FACTORY_MWPM
    assert "b0_factory_mwpm" in ladder.list_rungs()
    summ = ladder.summary()
    assert summ["total_arms"] >= 4
    assert "BaselineLadder" in repr(ladder)


def test_5_level_baseline_hierarchy_registry():
    """Verify that the 5-level baseline ladder hierarchy is strictly formalized."""
    expected_levels = [
        BaselineLevel.LEVEL_0_STATIC_MWPM,
        BaselineLevel.LEVEL_1_PERIODIC_SPITZ,
        BaselineLevel.LEVEL_2_CUSUM_GATED,
        BaselineLevel.LEVEL_3_FULL_CG_CCADS,
        BaselineLevel.ORACLE_CEILING,
    ]
    for lvl in expected_levels:
        assert lvl in BASELINE_LEVEL_REGISTRY
        info = BASELINE_LEVEL_REGISTRY[lvl]
        assert info.level == lvl
        assert len(info.formal_name) > 0
        assert len(info.description) > 0

    assert not BASELINE_LEVEL_REGISTRY[BaselineLevel.LEVEL_0_STATIC_MWPM].is_adaptive
    assert BASELINE_LEVEL_REGISTRY[BaselineLevel.LEVEL_1_PERIODIC_SPITZ].is_adaptive
    assert BASELINE_LEVEL_REGISTRY[BaselineLevel.LEVEL_2_CUSUM_GATED].is_adaptive
    assert BASELINE_LEVEL_REGISTRY[BaselineLevel.LEVEL_3_FULL_CG_CCADS].is_adaptive
    assert BASELINE_LEVEL_REGISTRY[BaselineLevel.ORACLE_CEILING].is_oracle


def test_compute_baseline_deltas_relative_benchmarks():
    """Verify relative performance delta computation against Level 0."""
    raw_results = {
        BaselineLevel.LEVEL_0_STATIC_MWPM: 0.0120,
        BaselineLevel.LEVEL_1_PERIODIC_SPITZ: 0.0150,  # +25% worse (variance explosion)
        BaselineLevel.LEVEL_2_CUSUM_GATED: 0.0115,     # ~4.17% improvement
        BaselineLevel.LEVEL_3_FULL_CG_CCADS: 0.0090,   # 25% improvement
        BaselineLevel.ORACLE_CEILING: 0.0072,          # 40% improvement (ceiling)
    }

    benchmarks = compute_baseline_deltas(raw_results)

    # Check Level 0 baseline benchmark
    b0 = benchmarks["level_0_static_mwpm"]
    assert b0.ler == 0.0120
    assert b0.delta_ler == 0.0
    assert b0.relative_delta_pct == 0.0
    assert b0.relative_reduction_pct == 0.0
    assert not b0.is_improvement

    # Check Level 1 (Periodic Spitz)
    b1 = benchmarks["level_1_periodic_spitz"]
    assert b1.ler == 0.0150
    assert b1.delta_ler == pytest.approx(0.0030)
    assert b1.relative_delta_pct == pytest.approx(25.0)
    assert b1.relative_reduction_pct == pytest.approx(-25.0)
    assert not b1.is_improvement

    # Check Level 3 (Full CG-CCADS)
    b3 = benchmarks["level_3_full_cg_ccads"]
    assert b3.ler == 0.0090
    assert b3.delta_ler == pytest.approx(-0.0030)
    assert b3.relative_delta_pct == pytest.approx(-25.0)
    assert b3.relative_reduction_pct == pytest.approx(25.0)
    assert b3.is_improvement

    # Check Level 4 (Oracle Ceiling)
    b4 = benchmarks["oracle_ceiling"]
    assert b4.ler == 0.0072
    assert b4.relative_reduction_pct == pytest.approx(40.0)
    assert b4.is_improvement


def test_compute_baseline_deltas_with_nested_dicts_and_string_keys():
    """Verify that compute_baseline_deltas handles dict entries and string keys."""
    results = {
        "static_mwpm": {"mean_ler": 0.010},
        "level_1": {"mean_ler": 0.012},
        "full_cg_ccads": {"mean_ler": 0.008},
        "oracle": {"mean_ler": 0.006},
    }
    benchmarks = compute_baseline_deltas(results, reference_level="static_mwpm")
    assert "level_0_static_mwpm" in benchmarks
    assert "level_1_periodic_spitz" in benchmarks
    assert "level_3_full_cg_ccads" in benchmarks
    assert "oracle_ceiling" in benchmarks

    cg_bench = benchmarks["level_3_full_cg_ccads"]
    assert cg_bench.relative_reduction_pct == pytest.approx(20.0)
    assert cg_bench.is_improvement


def test_baseline_ladder_level_methods():
    """Verify register_level, get_level_arm, and compare_deltas on BaselineLadder."""
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3)
    ladder = BaselineLadder(circuit=circuit)

    lvl0_arm = ladder.get_level_arm(BaselineLevel.LEVEL_0_STATIC_MWPM)
    assert lvl0_arm is not None
    assert lvl0_arm.level == BaselineLevel.LEVEL_0_STATIC_MWPM

    # Register Level 3 dynamically
    dummy_decoder = MWPMDecoder()
    dummy_decoder.configure(circuit=circuit)
    ladder.register_level(BaselineLevel.LEVEL_3_FULL_CG_CCADS, dummy_decoder)

    lvl3_arm = ladder.get_level_arm(BaselineLevel.LEVEL_3_FULL_CG_CCADS)
    assert lvl3_arm is not None
    assert lvl3_arm.level == BaselineLevel.LEVEL_3_FULL_CG_CCADS
    assert lvl3_arm.decoder == dummy_decoder

    # Test compare_deltas on the ladder instance
    deltas = ladder.compare_deltas({0: 0.020, 3: 0.016})
    assert deltas["level_3_full_cg_ccads"].relative_reduction_pct == pytest.approx(20.0)
