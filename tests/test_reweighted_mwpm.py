"""
Tests for OnlineReweightedMWPM (B3: Bhardwaj et al. PRX Quantum 2026 / DGR style).
"""

import numpy as np
import pytest
import stim

from adaptive_qec.decoders.reweighted_mwpm import OnlineReweightedMWPM
from adaptive_qec.decoders.registry import get_decoder


def test_reweighted_mwpm_initialization_and_configuration():
    """Verify that OnlineReweightedMWPM configures properly from circuit."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.005,
    )
    decoder = OnlineReweightedMWPM(window_size=100, update_interval=25)
    decoder.configure(circuit=circuit)

    assert decoder.name == "reweighted_mwpm"
    assert decoder._num_detectors > 0
    assert decoder._num_observables == 1


def test_reweighted_mwpm_decodes_and_updates():
    """Verify that streaming syndromes triggers weight updates and valid decoding."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.008,
    )
    sampler = circuit.compile_detector_sampler(seed=42)
    syndromes, observables = sampler.sample(shots=80, separate_observables=True)

    decoder = OnlineReweightedMWPM(window_size=50, update_interval=20)
    decoder.configure(circuit=circuit)

    # Initial decode batch
    metrics = decoder.decode_batch(syndromes, observables)
    assert metrics.total_shots == 80
    assert 0.0 <= metrics.logical_error_rate <= 1.0
    assert metrics.extra["total_reweights"] > 0


def test_registry_resolves_reweighted_mwpm():
    """Verify that get_decoder resolves 'reweighted_mwpm' and 'rw_mwpm'."""
    dec1 = get_decoder("reweighted_mwpm")
    dec2 = get_decoder("rw_mwpm")
    assert isinstance(dec1, OnlineReweightedMWPM)
    assert isinstance(dec2, OnlineReweightedMWPM)


def test_reweighted_mwpm_edge_cases_and_reset():
    """Verify empty windows, zero defects, unconfigured error, and reset behavior."""
    decoder = OnlineReweightedMWPM()
    assert not decoder.is_configured
    assert "configured=False" in repr(decoder)

    with pytest.raises(RuntimeError):
        decoder.decode(np.zeros(5, dtype=np.uint8))

    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=2)
    decoder.configure(circuit=circuit)
    assert decoder.is_configured

    # Empty window update
    decoder.update_weights(np.empty((0, decoder._num_detectors), dtype=np.uint8))
    assert decoder.total_reweights == 0

    # Empty batch decode
    empty_corr = decoder.decode(np.empty((0, decoder._num_detectors), dtype=np.uint8))
    assert empty_corr.observable_corrections.shape == (0, 1)

    empty_metrics = decoder.decode_batch(
        np.empty((0, decoder._num_detectors), dtype=np.uint8),
        np.empty((0, 1), dtype=np.uint8),
    )
    assert empty_metrics.total_shots == 0

    # All zero defects
    zeros = np.zeros((30, decoder._num_detectors), dtype=np.uint8)
    corr_zeros = decoder.decode(zeros)
    assert corr_zeros.observable_corrections.shape == (30, 1)

    # Test reset
    decoder.reset()
    assert len(decoder._syndrome_buffer) == 0
    assert decoder._shots_since_update == 0
    assert decoder.total_reweights == 0
