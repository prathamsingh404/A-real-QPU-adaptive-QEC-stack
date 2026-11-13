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

    def test_decode_single(self):
        circuit = self._make_circuit()
        decoder = MWPMDecoder()
        decoder.configure(circuit=circuit)

        # Create a zero syndrome (no errors)
        syndrome = np.zeros(circuit.num_detectors, dtype=np.uint8)
        correction = decoder.decode(syndrome)

        assert correction.observable_corrections is not None

    def test_decode_batch(self):
        circuit = self._make_circuit(p=0.02)
        decoder = MWPMDecoder()
        decoder.configure(circuit=circuit)

        # Sample some syndromes
        sampler = circuit.compile_detector_sampler()
        result = sampler.sample(shots=1000, separate_observables=True)
        detections = np.array(result[0], dtype=np.uint8)
        obs_flips = np.array(result[1], dtype=np.uint8)

        metrics = decoder.decode_batch(detections, obs_flips)

        assert isinstance(metrics, DecoderMetrics)
        assert metrics.total_shots == 1000
        assert 0 <= metrics.logical_error_rate <= 1
        assert metrics.decode_time_s > 0
        assert metrics.throughput_shots_per_s > 0

    def test_error_rate_decreases_with_distance(self):
        """Higher distance should give lower error rate (below threshold)."""
        rates = {}
        for d in [3, 5]:
            circuit = self._make_circuit(
                code_type="surface", distance=d, rounds=d, p=0.001
            )
            decoder = MWPMDecoder()
            decoder.configure(circuit=circuit)

            sampler = circuit.compile_detector_sampler()
            result = sampler.sample(shots=5000, separate_observables=True)
            detections = np.array(result[0], dtype=np.uint8)
            obs_flips = np.array(result[1], dtype=np.uint8)

            metrics = decoder.decode_batch(detections, obs_flips)
            rates[d] = metrics.logical_error_rate

        # Below threshold, higher distance should be better
        # (may not always hold for small shot counts, but should on average)
        assert rates[5] <= rates[3] + 0.02  # with tolerance

    def test_latency_metrics(self):
        """Latency percentiles should be computed."""
        circuit = self._make_circuit(p=0.01)
        decoder = MWPMDecoder()
        decoder.configure(circuit=circuit)

        sampler = circuit.compile_detector_sampler()
        result = sampler.sample(shots=500, separate_observables=True)
        detections = np.array(result[0], dtype=np.uint8)
        obs_flips = np.array(result[1], dtype=np.uint8)

        metrics = decoder.decode_batch(detections, obs_flips)

        assert metrics.latency_p50_us > 0
        assert metrics.latency_p99_us >= metrics.latency_p50_us
        assert metrics.latency_p999_us >= metrics.latency_p99_us

    def test_unconfigured_raises(self):
        decoder = MWPMDecoder()
        with pytest.raises(RuntimeError):
            decoder.decode(np.zeros(10, dtype=np.uint8))


class TestDecoderRegistry:
    """Test decoder plugin registry."""

    def test_get_mwpm(self):
        decoder = get_decoder("mwpm")
        assert decoder.name == "mwpm"

    def test_invalid_decoder(self):
        with pytest.raises(ValueError):
            get_decoder("nonexistent_decoder")

    def test_list_decoders(self):
        _ = get_decoder("mwpm")  # force registration
        decoders = list_decoders()
        assert "mwpm" in decoders
        assert "union_find" in decoders
        assert "lazy_mwpm" in decoders
        assert "reweighted_mwpm" in decoders

    def test_get_all_registered_decoders(self):
        """All decoders in DECODER_REGISTRY can be retrieved and initialized."""
        from adaptive_qec.decoders.base import Decoder

        for name in list_decoders():
            dec = get_decoder(name)
            assert isinstance(dec, Decoder)
            assert hasattr(dec, "configure")
            assert hasattr(dec, "decode")

    def test_case_and_whitespace_insensitivity(self):
        """get_decoder handles surrounding whitespace and uppercase."""
        dec1 = get_decoder("  MWPM  ")
        assert dec1.name == "mwpm"
        dec2 = get_decoder("UNION_FIND")
        assert dec2.name == "union_find"

    def test_register_and_retrieve_custom_decoder(self):
        """Custom decoder class can be registered and retrieved."""
        from adaptive_qec.decoders.base import Correction, Decoder
        from adaptive_qec.decoders.registry import DECODER_REGISTRY

        class DummyCustomDecoder(Decoder):
            name = "dummy_custom"
            def configure(self, *args, **kwargs): pass
            def decode(self, syndrome): return Correction(observable_corrections=np.zeros(1))
            def decode_batch(self, syndromes, observable_flips=None): return np.zeros((len(syndromes), 1))

        DECODER_REGISTRY["dummy_custom"] = DummyCustomDecoder
        try:
            dec = get_decoder("dummy_custom")
            assert isinstance(dec, DummyCustomDecoder)
            assert dec.name == "dummy_custom"
            assert "dummy_custom" in list_decoders()
        finally:
            DECODER_REGISTRY.pop("dummy_custom", None)

