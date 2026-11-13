"""
Tests for true quantum codes, dynamic feedforward, and multi-round decay analysis.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit

from adaptive_qec.qec.quantum_codes import (
    build_422_code_circuit,
    build_dynamic_feedforward_circuit,
    build_multi_round_circuit,
    fit_logical_error_decay,
)


class TestQuantumCodes:
    """Test [[4, 2, 2]] code and dynamic feedforward circuit generation."""

    def test_build_422_circuit(self):
        qc = build_422_code_circuit(insert_dd=True)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 6  # 4 data + 2 ancilla
        assert qc.num_clbits == 6  # 2 syn + 4 data
        ops = dict(qc.count_ops())
        assert "cx" in ops
        assert "measure" in ops
        assert "reset" in ops

    def test_build_422_circuit_without_dd(self):
        qc = build_422_code_circuit(insert_dd=False)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 6
        ops = dict(qc.count_ops())
        assert "x" not in ops  # No DD pulses inserted

    def test_build_dynamic_feedforward_circuit(self):
        qc = build_dynamic_feedforward_circuit()
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 3  # 2 data + 1 ancilla
        assert qc.num_clbits == 3  # 1 syn + 2 data
        ops = dict(qc.count_ops())
        assert "if_else" in ops or "if_test" in str(qc.data)

    def test_build_multi_round_circuit(self):
        qc = build_multi_round_circuit(distance=3, rounds=4, insert_dd=True)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 5  # 3 data + 2 ancilla
        # 4 rounds of 2-bit syndromes + 3-bit data = 11 clbits
        assert qc.num_clbits == 11

    def test_fit_logical_error_decay(self):
        rounds = [1, 2, 4, 8]
        # Synthetic exponential decay with A = 0.99, eps_L = 0.02
        true_A = 0.99
        true_eps = 0.02
        fidelities = [true_A * (1.0 - 2.0 * true_eps) ** r for r in rounds]

        fit = fit_logical_error_decay(rounds, fidelities)
        assert "logical_error_per_round" in fit
        assert "state_preparation_fidelity" in fit
        assert np.isclose(fit["state_preparation_fidelity"], true_A, atol=0.01)
        assert np.isclose(fit["logical_error_per_round"], true_eps, atol=0.005)


class TestQuantumCodeHierarchy:
    """Tests for QuantumCode class hierarchy, stabilizers, logical operators, and coordinate mappings."""

    def test_repetition_code_properties(self):
        from adaptive_qec.qec.quantum_codes import QuantumCode, RepetitionCode

        code = RepetitionCode(distance=3, rounds=2)
        assert isinstance(code, QuantumCode)
        assert len(code.stabilizer_generators) == 2
        assert code.stabilizers == code.stabilizer_generators
        assert "Z0 Z1" in code.stabilizer_generators
        assert "Z1 Z2" in code.stabilizer_generators

        # Logical operators
        ops = code.logical_operators
        assert "X" in ops and "Z" in ops
        assert ops["Z"] == "Z0"
        assert ops["X"] == "X0 X1 X2"
        assert code.logical_x == "X0 X1 X2"
        assert code.logical_z == "Z0"

        # Coordinate mappings
        coords = code.coordinate_mappings
        assert len(coords) == 5  # 3 data + 2 ancilla
        assert coords[0] == (0.0, 0.0)
        assert coords[1] == (2.0, 0.0)
        assert coords[2] == (4.0, 0.0)
        assert coords[3] == (1.0, 0.0)
        assert coords[4] == (3.0, 0.0)

    def test_surface_code_properties(self):
        from adaptive_qec.qec.quantum_codes import QuantumCode, SurfaceCode

        code = SurfaceCode(distance=3, rounds=2)
        assert isinstance(code, QuantumCode)

        stabs = code.stabilizer_generators
        assert len(stabs) == 8  # d^2 - 1 = 8 stabilizers for d=3
        x_stabs = [s for s in stabs if s.startswith("X")]
        z_stabs = [s for s in stabs if s.startswith("Z")]
        assert len(x_stabs) == 4
        assert len(z_stabs) == 4

        # Logical operators
        ops = code.logical_operators
        assert "X" in ops and "Z" in ops
        assert len(ops["X"].split()) == 3
        assert len(ops["Z"].split()) == 3

        # Coordinate mappings
        coords = code.coordinate_mappings
        assert len(coords) > 9
        for q, (x, y) in coords.items():
            assert isinstance(x, float)
            assert isinstance(y, float)

    def test_color_code_properties(self):
        from adaptive_qec.qec.quantum_codes import ColorCode, QuantumCode

        code = ColorCode(distance=3, rounds=2)
        assert isinstance(code, QuantumCode)
        assert code.num_data == 7
        assert code.num_ancilla == 3

        stabs = code.stabilizer_generators
        assert len(stabs) == 6  # 3 faces * (1 X + 1 Z)
        x_stabs = [s for s in stabs if s.startswith("X")]
        z_stabs = [s for s in stabs if s.startswith("Z")]
        assert len(x_stabs) == 3
        assert len(z_stabs) == 3

        ops = code.logical_operators
        assert "X" in ops and "Z" in ops

        coords = code.coordinate_mappings
        assert len(coords) == 10  # 7 data + 3 ancilla

        # Circuit generation & detector coordinates
        c = code.generate_circuit()
        assert c.num_detectors > 0
        det_coords = code.get_detector_coordinates()
        assert det_coords.shape[0] == c.num_detectors

    def test_color_code_validation(self):
        from adaptive_qec.qec.quantum_codes import ColorCode

        with pytest.raises(ValueError):
            ColorCode(distance=2, rounds=2)
        with pytest.raises(ValueError):
            ColorCode(distance=3, rounds=1)

    def test_create_code_factory_with_color(self):
        from adaptive_qec.qec.quantum_codes import ColorCode, create_code

        code = create_code("color", distance=3, rounds=2)
        assert isinstance(code, ColorCode)

