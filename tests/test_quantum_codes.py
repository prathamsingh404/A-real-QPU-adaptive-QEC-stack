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
