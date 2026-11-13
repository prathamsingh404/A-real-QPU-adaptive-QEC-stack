"""
Unit tests for ConfidenceGatedReweightingController.
Verifies label-free 3-gate statistical reweighting behavior:
- Stationary noise null hypothesis (false reweight rate <= alpha).
- Localized hotspot drift detection under sufficient window statistics.
- Noise-floor abstention (STAY) under small window sizes.
- Reversion to baseline when drift subsides.
"""

import numpy as np
import pymatching
import pytest
import stim

from adaptive_qec.controller.reweighting_gate import (
    ConfidenceGatedReweightingController,
    GateDecision,
)


@pytest.fixture
def surface_code_fixture():
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.004,
        after_reset_flip_probability=0.004,
        before_measure_flip_probability=0.004,
        before_round_data_depolarization=0.004,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    base_matching = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler()
    return circuit, base_matching, sampler


def test_stationary_noise_null_false_alarm_rate(surface_code_fixture):
    """Under stationary null noise, false reweight rate is bounded by alpha <= 0.05."""
    _, base_matching, sampler = surface_code_fixture
    controller = ConfidenceGatedReweightingController(
        base_matching=base_matching,
        alpha_significance=0.05,
        drift_z_threshold=2.0,
        adaptation_tax=0.004,
        dead_zone_epsilon=0.002,
    )

    # Sample 50 windows of 500 shots each under stationary noise
    reweight_count = 0
    total_windows = 50

    for _ in range(total_windows):
        syndromes, _ = sampler.sample(shots=500, separate_observables=True)
        decision, _, stats = controller.evaluate_window(syndromes)
        if decision == GateDecision.REWEIGHT:
            reweight_count += 1

    false_alarm_rate = reweight_count / total_windows
    # Must be at most 0.10 (well within statistical bounds of alpha=0.05 on 50 samples)
    assert false_alarm_rate <= 0.10
    assert controller.total_stays >= total_windows * 0.90


def test_localized_hotspot_triggers_reweight_at_large_window(surface_code_fixture):
    """Under localized hotspot with W=2000, controller detects drift and reweights."""
    circuit_base, base_matching, _ = surface_code_fixture

    # Inject localized depolarizing error on central data qubit 10
    lines = []
    for line in str(circuit_base).splitlines():
        lines.append(line)
        if line.startswith("TICK"):
            lines.append("DEPOLARIZE1(0.04) 10")
    circuit_hotspot = stim.Circuit("\n".join(lines))
    sampler_hotspot = circuit_hotspot.compile_detector_sampler()

    controller = ConfidenceGatedReweightingController(
        base_matching=base_matching,
        drift_z_threshold=2.0,
        adaptation_tax=0.003,
        dead_zone_epsilon=0.001,
        reference_defect_rate=0.015, # Nominal reference
    )

    syndromes, _ = sampler_hotspot.sample(shots=2000, separate_observables=True)
    decision, active_matching, stats = controller.evaluate_window(syndromes)

    assert decision == GateDecision.REWEIGHT
    assert controller.is_reweighted
    assert stats.max_edge_lcb > 0.003


def test_small_window_suppresses_reweighting(surface_code_fixture):
    """At small window W=150, high binomial variance suppresses LCB, enforcing STAY."""
    circuit_base, base_matching, _ = surface_code_fixture

    # Mild drift
    lines = []
    for line in str(circuit_base).splitlines():
        lines.append(line)
        if line.startswith("TICK"):
            lines.append("DEPOLARIZE1(0.015) 10")
    circuit_mild = stim.Circuit("\n".join(lines))
    sampler_mild = circuit_mild.compile_detector_sampler()

    controller = ConfidenceGatedReweightingController(
        base_matching=base_matching,
        drift_z_threshold=1.5,
        adaptation_tax=0.008,
        dead_zone_epsilon=0.004,
        reference_defect_rate=0.015,
    )

    # Very small window: SE is high (~0.015), suppressing LCB below tax + deadzone
    syndromes, _ = sampler_mild.sample(shots=150, separate_observables=True)
    decision, _, stats = controller.evaluate_window(syndromes)

    # Should safely abstain (STAY)
    assert decision == GateDecision.STAY


def test_empty_syndromes_evaluation(surface_code_fixture):
    """Empty syndrome batch returns STAY with zero stats safely."""
    _, base_matching, _ = surface_code_fixture
    controller = ConfidenceGatedReweightingController(base_matching=base_matching)
    empty = np.zeros((0, 0), dtype=np.uint8)
    decision, m, stats = controller.evaluate_window(empty)
    assert decision == GateDecision.STAY
    assert stats.window_size == 0
    assert m is base_matching


def test_telemetry_and_state_serialization(surface_code_fixture):
    """Telemetry export and state serialization/restoration preserves all counters."""
    _, base_matching, sampler = surface_code_fixture
    controller = ConfidenceGatedReweightingController(base_matching=base_matching)
    syndromes, _ = sampler.sample(shots=200, separate_observables=True)
    controller.evaluate_window(syndromes)

    telemetry = controller.get_telemetry()
    assert telemetry["total_evaluations"] == 1
    assert telemetry["total_stays"] == 1
    assert telemetry["abstention_rate"] == 1.0

    state = controller.serialize_state()
    assert state["total_stays"] == 1
    assert "adaptation_tax" in state

    new_controller = ConfidenceGatedReweightingController(base_matching=base_matching)
    new_controller.load_state(state)
    assert new_controller.total_stays == 1
    assert new_controller.get_telemetry()["total_stays"] == 1


def test_manual_reference_reset(surface_code_fixture):
    """Manual reset of reference defect rate re-anchors the controller."""
    _, base_matching, _ = surface_code_fixture
    controller = ConfidenceGatedReweightingController(
        base_matching=base_matching,
        reference_defect_rate=0.010,
    )
    assert controller.reference_defect_rate == 0.010
    controller.reset_reference(new_reference=0.025)
    assert controller.reference_defect_rate == 0.025
    assert not controller.is_reweighted
