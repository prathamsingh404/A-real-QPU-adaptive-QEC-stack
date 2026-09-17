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
