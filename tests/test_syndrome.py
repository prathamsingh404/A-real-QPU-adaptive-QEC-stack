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

    def test_noiseless_circuit_no_detections(self):
        """A noiseless circuit should produce no detection events."""
        code = RepetitionCode(distance=3, rounds=3)
        circuit = code.generate_circuit(noise=None)
        extractor = SyndromeExtractor(circuit)

        detections, obs = extractor.sample_detectors(shots=100)

        # All zeros — no errors means no detections
        assert detections.sum() == 0
        assert obs.sum() == 0

    def test_noisy_circuit_has_detections(self):
        """A noisy circuit should produce some detection events."""
        circuit = self._make_noisy_circuit(p=0.05)
        extractor = SyndromeExtractor(circuit)

        detections, obs = extractor.sample_detectors(shots=1000)

        assert detections.sum() > 0  # should have some detections

    def test_syndrome_tensor_shape(self):
        """Test syndrome tensor reshaping."""
        circuit = self._make_noisy_circuit(rounds=5)
        extractor = SyndromeExtractor(circuit)

        detections, obs = extractor.sample_detectors(shots=100)
        tensor = extractor.build_syndrome_tensor(detections, num_rounds=5)

        assert tensor.ndim == 3
        assert tensor.shape[0] == 100  # shots

    def test_detector_graph(self):
        """Test detector graph construction."""
        circuit = self._make_noisy_circuit(p=0.01)
        extractor = SyndromeExtractor(circuit)

        graph = extractor.build_detector_graph()

        assert graph.num_detectors == circuit.num_detectors
        assert len(graph.edges) > 0  # should have error mechanisms

    def test_surface_code_extraction(self):
        """Test extraction for surface code."""
        circuit = self._make_noisy_circuit(code_type="surface", distance=3, rounds=3)
        extractor = SyndromeExtractor(circuit)

        detections, obs = extractor.sample_detectors(shots=500)

        assert detections.shape[0] == 500
        assert detections.shape[1] == circuit.num_detectors

    def test_detector_record_creation(self):
        """Test creating a DetectorRecord."""
        circuit = self._make_noisy_circuit()
        extractor = SyndromeExtractor(circuit)

        detections, obs = extractor.sample_detectors(shots=100)
        record = extractor.create_detector_record(
            experiment_id="test_001",
            detection_events=detections,
            observable_flips=obs,
            num_rounds=3,
        )

        assert record.experiment_id == "test_001"
        assert record.num_detectors == circuit.num_detectors
