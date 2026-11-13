"""
End-to-End Online Adaptation Pipeline Integration Tests (tests/test_online_adaptation_pipeline.py).

Verifies the closed-loop online pipeline:
1. Circuit construction with Stim
2. Baseline PyMatching decoder initialization
3. Syndrome sampling across stationary and drifted regimes
4. Label-free gating decisions via ConfidenceGatedReweightingController
5. Dynamic decoder update and batch syndrome decoding
"""

import numpy as np
import pytest
import pymatching
import stim

from adaptive_qec.controller.reweighting_gate import (
    ConfidenceGatedReweightingController,
    GateDecision,
)


def make_surface_code(distance: int = 3, rounds: int = 3, p: float = 0.005) -> stim.Circuit:
    """Generate a standard rotated memory Z surface code circuit."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
    )


def test_end_to_end_online_pipeline_stationary_abstains():
    """Verify stationary noise leads to STAY decision, keeping base matching graph unchanged."""
    circuit = make_surface_code(distance=3, rounds=3, p=0.004)
    dem = circuit.detector_error_model(decompose_errors=True)
    base_matching = pymatching.Matching.from_detector_error_model(dem)

    sampler = circuit.compile_detector_sampler()
    # Baseline sample to establish reference defect rate
    baseline_syndromes, _ = sampler.sample(shots=500, separate_observables=True)
    ref_rate = float(baseline_syndromes.mean())

    controller = ConfidenceGatedReweightingController(
        base_matching=base_matching,
        reference_defect_rate=ref_rate,
        drift_z_threshold=2.5,
        adaptation_tax=0.003,
    )

    # Process stationary window
    eval_syndromes, eval_obs = sampler.sample(shots=200, separate_observables=True)
    decision, active_matching, stats = controller.evaluate_window(eval_syndromes)

    assert decision == GateDecision.STAY
    assert not controller.is_reweighted
    assert controller.total_stays == 1
    assert controller.total_reweights == 0

    # Test decoding with active matching
    predictions = active_matching.decode_batch(eval_syndromes)
    assert predictions.shape == eval_obs.shape
    ler = np.mean(predictions[:, 0] != eval_obs[:, 0])
    assert ler < 0.15  # Low LER for d=3 at p=0.004


def test_end_to_end_online_pipeline_drift_triggers_reweight():
    """Verify severe localized physical drift triggers REWEIGHT and updates matching graph."""
    nominal_circuit = make_surface_code(distance=3, rounds=3, p=0.003)
    dem = nominal_circuit.detector_error_model(decompose_errors=True)
    base_matching = pymatching.Matching.from_detector_error_model(dem)

    sampler_nominal = nominal_circuit.compile_detector_sampler()
    baseline_syndromes, _ = sampler_nominal.sample(shots=1000, separate_observables=True)
    ref_rate = float(baseline_syndromes.mean())

    controller = ConfidenceGatedReweightingController(
        base_matching=base_matching,
        reference_defect_rate=ref_rate,
        drift_z_threshold=2.5,
        adaptation_tax=0.001,
        dead_zone_epsilon=0.001,
    )

    # Severe drift circuit (10x noise)
    drifted_circuit = make_surface_code(distance=3, rounds=3, p=0.035)
    sampler_drifted = drifted_circuit.compile_detector_sampler()
    drifted_syndromes, drifted_obs = sampler_drifted.sample(shots=500, separate_observables=True)

    decision, active_matching, stats = controller.evaluate_window(drifted_syndromes)

    assert decision == GateDecision.REWEIGHT
    assert controller.is_reweighted
    assert controller.total_reweights == 1
    assert stats.drift_score_z > 2.5

    # Verify active matching can decode drifted syndromes
    predictions = active_matching.decode_batch(drifted_syndromes)
    assert predictions.shape == drifted_obs.shape


def test_end_to_end_online_pipeline_multi_window_lifecycle():
    """Verify full lifecycle: stationary -> stay, drift -> reweight, subside -> reset."""
    nominal_circuit = make_surface_code(distance=3, rounds=3, p=0.004)
    dem = nominal_circuit.detector_error_model(decompose_errors=True)
    base_matching = pymatching.Matching.from_detector_error_model(dem)

    sampler_nominal = nominal_circuit.compile_detector_sampler()
    baseline_syndromes, _ = sampler_nominal.sample(shots=1000, separate_observables=True)
    ref_rate = float(baseline_syndromes.mean())

    controller = ConfidenceGatedReweightingController(
        base_matching=base_matching,
        reference_defect_rate=ref_rate,
        drift_z_threshold=2.5,
        adaptation_tax=0.001,
        dead_zone_epsilon=0.001,
    )

    # Window 1: Stationary -> STAY
    w1, _ = sampler_nominal.sample(shots=200, separate_observables=True)
    d1, _, _ = controller.evaluate_window(w1)
    assert d1 == GateDecision.STAY

    # Window 2: Severe drift -> REWEIGHT
    drifted_circuit = make_surface_code(distance=3, rounds=3, p=0.04)
    sampler_drifted = drifted_circuit.compile_detector_sampler()
    w2, _ = sampler_drifted.sample(shots=500, separate_observables=True)
    d2, _, _ = controller.evaluate_window(w2)
    assert d2 == GateDecision.REWEIGHT
    assert controller.is_reweighted

    # Window 3: Drift subsides back to nominal -> RESET
    w3, _ = sampler_nominal.sample(shots=500, separate_observables=True)
    d3, _, _ = controller.evaluate_window(w3)
    assert d3 == GateDecision.RESET
    assert not controller.is_reweighted

    assert controller.total_stays == 1
    assert controller.total_reweights == 1
    assert controller.total_resets == 1
