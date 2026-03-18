"""Tests for MWPM decoder and decoder framework."""

import numpy as np
import pytest
import stim

from adaptive_qec.config import NoiseConfig
from adaptive_qec.decoders.base import DecoderMetrics
from adaptive_qec.decoders.mwpm import MWPMDecoder
from adaptive_qec.decoders.registry import get_decoder, list_decoders
from adaptive_qec.qec.codes import RepetitionCode, SurfaceCode


class TestMWPMDecoder:
    """Test MWPM decoder with PyMatching."""

    def _make_circuit(self, code_type="repetition", distance=3, rounds=3, p=0.01):
        noise = NoiseConfig()
        noise.gate.two_qubit = p

        if code_type == "repetition":
            code = RepetitionCode(distance=distance, rounds=rounds)
        else:
            code = SurfaceCode(distance=distance, rounds=rounds)

        return code.generate_circuit(noise=noise)

    def test_configure_from_circuit(self):
        circuit = self._make_circuit()
        decoder = MWPMDecoder()
        decoder.configure(circuit=circuit)

        assert decoder.name == "mwpm"

    def test_configure_from_dem(self):
        circuit = self._make_circuit()
        dem = circuit.detector_error_model(decompose_errors=True)

        decoder = MWPMDecoder()
        decoder.configure(dem=dem)
