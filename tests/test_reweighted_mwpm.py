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
