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

    def test_create_unrotated_surface_and_color(self):
        from adaptive_qec.qec.codes import UnrotatedSurfaceCode, ColorCode
        u_code = create_code("unrotated_surface", distance=3, rounds=2)
        assert isinstance(u_code, UnrotatedSurfaceCode)
        c_u = u_code.generate_circuit()
        assert c_u.num_detectors > 0

        color_code = create_code("color", distance=3, rounds=2)
        assert isinstance(color_code, ColorCode)
        c_color = color_code.generate_circuit()
        assert c_color.num_detectors > 0


    def test_invalid_type(self):
        with pytest.raises(ValueError):
            create_code("invalid_code", distance=3, rounds=3)

    def test_build_repetition_code_circuit_utility(self):
        from adaptive_qec.qec.circuits import build_repetition_code_circuit
        c = build_repetition_code_circuit(distance=4, rounds=3, noise_rate=0.002)
        assert isinstance(c, stim.Circuit)
        assert c.num_detectors > 0
        assert c.num_observables == 1


class TestCircuitBuildersAndScheduling:
    """Tests for enhanced circuit builders and customizable stabilizer scheduling."""

    def test_build_repetition_code_memory(self):
        from adaptive_qec.qec.circuits import build_repetition_code_memory

        circuit = build_repetition_code_memory(distance=3, rounds=4, p_error=0.005)
        assert isinstance(circuit, stim.Circuit)
        assert circuit.num_detectors > 0
        assert circuit.num_observables == 1

        dem = circuit.detector_error_model(decompose_errors=True)
        assert dem.num_detectors == circuit.num_detectors

        sampler = circuit.compile_detector_sampler()
        samples, obs = sampler.sample(shots=20, separate_observables=True)
        assert samples.shape == (20, circuit.num_detectors)
        assert obs.shape == (20, 1)

    def test_build_repetition_code_memory_validation(self):
        from adaptive_qec.qec.circuits import build_repetition_code_memory

        with pytest.raises(ValueError):
            build_repetition_code_memory(distance=2, rounds=2)
        with pytest.raises(ValueError):
            build_repetition_code_memory(distance=3, rounds=0)

    def test_build_rotated_surface_code_cx_vs_cz(self):
        from adaptive_qec.qec.circuits import build_rotated_surface_code

        c_cx = build_rotated_surface_code(distance=3, rounds=2, schedule="cx")
        c_cz = build_rotated_surface_code(distance=3, rounds=2, schedule="cz")
        c_pipe = build_rotated_surface_code(distance=3, rounds=2, schedule="pipelined_cz")

        # CX circuit should contain CX gates
        op_names_cx = [inst.name for inst in c_cx.flattened()]
        assert "CX" in op_names_cx
        assert "CZ" not in op_names_cx

        # CZ circuit should replace CX with CZ
        op_names_cz = [inst.name for inst in c_cz.flattened()]
        assert "CZ" in op_names_cz
        assert "CX" not in op_names_cz

        # Pipelined CZ should also contain CZ and TICK
        op_names_pipe = [inst.name for inst in c_pipe.flattened()]
        assert "CZ" in op_names_pipe
        assert "CX" not in op_names_pipe
        assert "TICK" in op_names_pipe

        # All schedules should preserve detector counts and produce valid DEMs
        assert c_cx.num_detectors == c_cz.num_detectors == c_pipe.num_detectors
        dem_cx = c_cx.detector_error_model(decompose_errors=True)
        dem_cz = c_cz.detector_error_model(decompose_errors=True)
        dem_pipe = c_pipe.detector_error_model(decompose_errors=True)
        assert dem_cx.num_detectors == dem_cz.num_detectors == dem_pipe.num_detectors

    def test_build_rotated_surface_code_validation(self):
        from adaptive_qec.qec.circuits import build_rotated_surface_code

        with pytest.raises(ValueError):
            build_rotated_surface_code(distance=4, rounds=2)  # Even distance
        with pytest.raises(ValueError):
            build_rotated_surface_code(distance=3, rounds=0)  # Invalid rounds
        with pytest.raises(ValueError):
            build_rotated_surface_code(distance=3, rounds=2, schedule="unsupported")

    def test_circuit_generator_scheduling_and_qiskit(self):
        from adaptive_qec.config import QECConfig
        from adaptive_qec.qec.circuits import CircuitGenerator

        config = QECConfig(distance=3, rounds=2)
        gen = CircuitGenerator(config, schedule="pipelined_cz")
        assert gen.schedule == "pipelined_cz"

        circuit = gen.generate_stim_circuit()
        op_names = [inst.name for inst in circuit.flattened()]
        assert "CZ" in op_names
        assert "CX" not in op_names

        # Convert to Qiskit
        qc = gen.stim_to_qiskit(circuit)
        ops = dict(qc.count_ops())
        assert "cz" in ops

