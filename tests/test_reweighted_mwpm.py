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
