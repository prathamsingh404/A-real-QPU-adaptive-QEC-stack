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
