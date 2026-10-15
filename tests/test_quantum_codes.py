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
