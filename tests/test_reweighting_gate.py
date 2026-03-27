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

