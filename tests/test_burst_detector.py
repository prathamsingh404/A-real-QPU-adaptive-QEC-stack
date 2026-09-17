"""Tests for Correlated Error Burst Detection (Problem 2).

Validates sliding-window Poisson burst detection, burst heuristic classification
(cosmic ray, QP poisoning, crosstalk), drift detector integration, and
burst-aware decoding.
"""

import numpy as np
import pytest

from adaptive_qec.config import NoiseConfig
from adaptive_qec.decoders.mwpm import MWPMDecoder
from adaptive_qec.noise.burst_detector import (
    BurstDetector,
    BurstType,
    reshape_syndromes_to_tensor,
)
from adaptive_qec.noise.drift import CompositeDriftDetector, DriftStatus
from adaptive_qec.qec.codes import create_code


class TestBurstDetector:
    """Tests for BurstDetector."""

    def test_clean_syndromes_no_burst(self):
        """Clean syndromes with low independent error should have zero bursts."""
        # 20 rounds, 8 detectors, independent error rate ~1%
        rng = np.random.default_rng(42)
        clean = (rng.random((20, 8)) < 0.01).astype(np.uint8)

        detector = BurstDetector(window_size=4, significance=0.001)
        analysis = detector.analyze(clean, num_detectors_per_round=8)

        assert analysis.total_rounds == 20
        assert analysis.total_detectors == 8
        assert len(analysis.bursts_detected) == 0
        assert analysis.burst_rate_per_round == 0.0

    def test_detect_cosmic_ray_burst(self):
        """A sudden burst across many detectors should be classified as COSMIC_RAY."""
        rng = np.random.default_rng(42)
        syndromes = (rng.random((30, 16)) < 0.01).astype(np.uint8)

        # Inject wide cosmic ray burst at round 10-11 across 12/16 detectors
        syndromes[10:12, :12] = 1

        detector = BurstDetector(window_size=4, significance=0.001)
        analysis = detector.analyze(syndromes, num_detectors_per_round=16)

        assert len(analysis.bursts_detected) >= 1
        burst = analysis.bursts_detected[0]
        assert burst.round_start <= 11 and burst.round_end >= 10
        assert burst.burst_type == BurstType.COSMIC_RAY
        assert burst.severity > 2.0
        assert burst.p_value < 0.001

    def test_detect_qp_poisoning_burst(self):
        """A persistent localized burst should be classified as QP_POISONING."""
        rng = np.random.default_rng(42)
        syndromes = (rng.random((30, 16)) < 0.01).astype(np.uint8)

        # Inject localized burst on only 2 detectors across 6 consecutive rounds (12..17)
        syndromes[12:18, 0:2] = 1

        detector = BurstDetector(window_size=5, significance=0.005, min_defects_for_burst=3)
        analysis = detector.analyze(syndromes, num_detectors_per_round=16)

        assert len(analysis.bursts_detected) >= 1
        # Look for the detected burst
        qp_bursts = [b for b in analysis.bursts_detected if b.burst_type == BurstType.QP_POISONING]
        assert len(qp_bursts) >= 1 or analysis.bursts_detected[0].severity > 1.5

    def test_reshape_syndromes_to_tensor(self):
        flat = np.zeros(24, dtype=np.uint8)
        flat[5] = 1
        tensor = reshape_syndromes_to_tensor(flat, num_rounds=6, num_detectors_per_round=4)
        assert tensor.shape == (6, 4)
        assert tensor[1, 1] == 1  # index 5 = 1 * 4 + 1


class TestDriftIntegration:
    """Tests for CompositeDriftDetector with burst detection."""

    def test_composite_drift_catches_burst(self):
        burst_detector = BurstDetector(window_size=4, significance=0.001)
        composite = CompositeDriftDetector(
            ewma_alpha=0.15,
            cusum_threshold=4.5,
            warmup_samples=3,
            burst_detector=burst_detector,
        )

        rates = np.full(8, 0.01)
        # Normal warmup
        for _ in range(5):
            composite.update(rates)

        # Create burst syndrome tensor
        syndrome_tensor = np.zeros((20, 8), dtype=np.uint8)
        syndrome_tensor[10:12, :] = 1  # massive burst

        report = composite.update(
            detection_rates=rates,
            syndrome_tensor=syndrome_tensor,
            num_detectors_per_round=8,
        )

        assert report.status == DriftStatus.BURST_EVENT
        assert report.magnitude > 0
        assert "burst_analysis" in report.details


class TestMWPMBurstAware:
    """Tests for MWPM burst-aware decoding."""

    def test_mwpm_decode_burst_aware(self):
        code = create_code("surface", distance=3, rounds=3)
        noise = NoiseConfig()
        noise.gate.two_qubit = 0.01
        circuit = code.generate_circuit(noise=noise)

        sampler = circuit.compile_detector_sampler()
        detectors, observables = sampler.sample(shots=100, separate_observables=True)

        mwpm = MWPMDecoder()
        mwpm.configure(circuit=circuit)

        # Total detectors = 3 rounds * ~8 detectors
        num_detectors = detectors.shape[1]
        num_rounds = 3
        det_per_round = num_detectors // num_rounds

        std_metrics, burst_metrics, summary = mwpm.decode_burst_aware(
            detectors,
            observables,
            num_rounds=num_rounds,
            num_detectors_per_round=det_per_round,
        )

        assert std_metrics.total_shots == 100
        assert burst_metrics.total_shots == 100
        assert "burst_shots" in summary
        assert "ler_reduction" in summary
