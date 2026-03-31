"""Tests for syndrome extraction."""

import numpy as np
import pytest
import stim

from adaptive_qec.config import NoiseConfig
from adaptive_qec.qec.codes import RepetitionCode, SurfaceCode
from adaptive_qec.syndrome.extraction import SyndromeExtractor


class TestSyndromeExtractor:
    """Test syndrome extraction from Stim circuits."""

    def _make_noisy_circuit(self, code_type="repetition", distance=3, rounds=3, p=0.01):
        """Helper to create a noisy circuit."""
        noise = NoiseConfig()
        noise.gate.two_qubit = p

        if code_type == "repetition":
            code = RepetitionCode(distance=distance, rounds=rounds)
        else:
            code = SurfaceCode(distance=distance, rounds=rounds)

        return code.generate_circuit(noise=noise)

    def test_stim_sampling(self):
        """Test direct Stim detector sampling."""
        circuit = self._make_noisy_circuit()
        extractor = SyndromeExtractor(circuit)

        detections, obs = extractor.sample_detectors(shots=1000)

        assert detections.shape[0] == 1000
        assert detections.shape[1] == circuit.num_detectors
        assert obs.shape[0] == 1000
        assert obs.shape[1] == circuit.num_observables
