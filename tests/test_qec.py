"""Tests for QEC code generation and circuit conversion."""

import numpy as np
import pytest
import stim

from adaptive_qec.config import NoiseConfig
from adaptive_qec.qec.codes import RepetitionCode, SurfaceCode, create_code


class TestRepetitionCode:
    """Test repetition code circuit generation."""

    def test_basic_generation(self):
        code = RepetitionCode(distance=3, rounds=3)
        circuit = code.generate_circuit()

        assert isinstance(circuit, stim.Circuit)
        assert circuit.num_detectors > 0
        assert circuit.num_observables == 1

    def test_info(self):
        code = RepetitionCode(distance=5, rounds=3)
        info = code.get_info()

        assert info.name == "repetition"
        assert info.distance == 5
        assert info.num_data_qubits == 5
        assert info.num_ancilla_qubits == 4

    def test_detector_count_scaling(self):
        """Detectors should scale with distance and rounds."""
        code3 = RepetitionCode(distance=3, rounds=3)
        code5 = RepetitionCode(distance=5, rounds=3)

        c3 = code3.generate_circuit()
        c5 = code5.generate_circuit()

        assert c5.num_detectors > c3.num_detectors

    def test_with_noise(self):
        code = RepetitionCode(distance=3, rounds=3)
        noise = NoiseConfig()
        circuit = code.generate_circuit(noise=noise)

        assert isinstance(circuit, stim.Circuit)

    def test_invalid_distance(self):
        with pytest.raises(ValueError):
            RepetitionCode(distance=2, rounds=3)

    def test_invalid_rounds(self):
        with pytest.raises(ValueError):
            RepetitionCode(distance=3, rounds=0)

    def test_detector_sampling(self):
        """Verify the circuit can be sampled correctly."""
        code = RepetitionCode(distance=3, rounds=3)
        noise = NoiseConfig()
        noise.gate.two_qubit = 0.01
        circuit = code.generate_circuit(noise=noise)

        sampler = circuit.compile_detector_sampler()
        result = sampler.sample(shots=100, separate_observables=True)

        detection_events = np.array(result[0])
        observable_flips = np.array(result[1])

        assert detection_events.shape[0] == 100
        assert detection_events.shape[1] == circuit.num_detectors
        assert observable_flips.shape[1] == 1


class TestSurfaceCode:
    """Test surface code circuit generation."""

    def test_basic_generation(self):
        code = SurfaceCode(distance=3, rounds=3)
        circuit = code.generate_circuit()

        assert isinstance(circuit, stim.Circuit)
        assert circuit.num_detectors > 0
        assert circuit.num_observables == 1

    def test_distance_scaling(self):
        """Higher distance should produce more qubits and detectors."""
        c3 = SurfaceCode(distance=3, rounds=3).generate_circuit()
        c5 = SurfaceCode(distance=5, rounds=3).generate_circuit()

        assert c5.num_qubits > c3.num_qubits
        assert c5.num_detectors > c3.num_detectors

    def test_with_noise(self):
        code = SurfaceCode(distance=3, rounds=3)
        noise = NoiseConfig()
        noise.gate.two_qubit = 0.01
        circuit = code.generate_circuit(noise=noise)

        assert isinstance(circuit, stim.Circuit)

    def test_detector_error_model(self):
        code = SurfaceCode(distance=3, rounds=3)
        noise = NoiseConfig()
        noise.gate.two_qubit = 0.01
        circuit = code.generate_circuit(noise=noise)

        dem = circuit.detector_error_model(decompose_errors=True)
        assert dem.num_detectors == circuit.num_detectors


class TestCodeFactory:
    """Test the create_code factory function."""

    def test_create_repetition(self):
        code = create_code("repetition", distance=3, rounds=3)
        assert isinstance(code, RepetitionCode)

    def test_create_surface(self):
        code = create_code("surface", distance=3, rounds=3)
        assert isinstance(code, SurfaceCode)

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            create_code("invalid_code", distance=3, rounds=3)
