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
