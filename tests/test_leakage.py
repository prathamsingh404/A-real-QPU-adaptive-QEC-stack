"""Tests for Leakage Detection and Reset Protocol (Problem 6).

Validates temporal autocorrelation, streak persistence analysis,
leakage candidate detection, rate estimation (γ_L, γ_S), and digital twin
leakage tracking.
"""

import numpy as np
import pytest

from adaptive_qec.digital_twin.twin import HardwareDigitalTwin
from adaptive_qec.noise.characterization import NoiseCharacterizer
from adaptive_qec.noise.leakage import (
    LeakageAnalysis,
    LeakageDetector,
    LeakageRateEstimator,
    LeakedQubit,
)


class TestLeakageDetector:
    """Tests for LeakageDetector."""

    def test_transient_noise_no_leakage(self):
        """Random independent errors should not trigger leakage detection."""
        rng = np.random.default_rng(42)
        # 30 rounds, 8 detectors, independent errors at 3%
        trace = (rng.random((30, 8)) < 0.03).astype(np.uint8)

        detector = LeakageDetector(min_persistence=4, autocorr_threshold=0.3)
        analysis = detector.analyze(trace)

