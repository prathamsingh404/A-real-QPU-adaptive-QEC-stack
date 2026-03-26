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
