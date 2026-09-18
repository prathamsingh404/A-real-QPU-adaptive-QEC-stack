"""Tests for noise statistics and drift detection."""

import numpy as np
import pytest

from adaptive_qec.noise.statistics import (
    compute_detector_statistics,
    compute_temporal_correlation,
    compute_spatial_correlation,
)
from adaptive_qec.noise.drift import (
    CompositeDriftDetector,
    CUSUMDriftDetector,
    DriftStatus,
    EWMADriftDetector,
)


class TestDetectorStatistics:
    """Test detector statistics computation."""

    def test_basic_statistics(self):
        rng = np.random.default_rng(42)
        # 1000 shots, 10 detectors with ~10% detection rate
        data = rng.binomial(1, 0.1, size=(1000, 10)).astype(np.uint8)

        stats = compute_detector_statistics(data)

        assert stats.num_detectors == 10
        assert stats.num_shots == 1000
        assert 0.05 < stats.mean_detection_rate < 0.15
        assert stats.detection_rates.shape == (10,)
        assert stats.pair_correlations.shape == (10, 10)

    def test_zero_rate_detectors(self):
        """Detectors with zero rate should not crash."""
        data = np.zeros((100, 5), dtype=np.uint8)
        stats = compute_detector_statistics(data)

        assert stats.mean_detection_rate == 0.0
        assert len(stats.hotspot_detectors) == 0

    def test_hotspot_detection(self):
        """A detector with unusually high rate should be flagged."""
        rng = np.random.default_rng(42)
        data = rng.binomial(1, 0.05, size=(10000, 10)).astype(np.uint8)
        # Make detector 3 a hotspot
        data[:, 3] = rng.binomial(1, 0.5, size=10000).astype(np.uint8)

        stats = compute_detector_statistics(data)
        assert 3 in stats.hotspot_detectors


class TestTemporalCorrelation:
    """Test temporal autocorrelation."""

    def test_basic_temporal(self):
        rng = np.random.default_rng(42)
        # 500 shots, 20 detectors (4 rounds × 5 per round)
        data = rng.binomial(1, 0.1, size=(500, 20)).astype(np.uint8)

        result = compute_temporal_correlation(data, num_rounds=3, max_lag=3)

        assert result.max_lag == 3
        assert result.mean_autocorrelation.shape == (3,)


class TestSpatialCorrelation:
    """Test spatial correlation analysis."""

    def test_basic_spatial(self):
        rng = np.random.default_rng(42)
        data = rng.binomial(1, 0.1, size=(1000, 10)).astype(np.uint8)

        result = compute_spatial_correlation(data)

        assert result.num_detectors == 10
        assert result.correlation_matrix.shape == (10, 10)


class TestEWMADriftDetector:
    """Test EWMA drift detection."""

    def test_stable_signal(self):
        """Stable signal should not trigger drift."""
        detector = EWMADriftDetector(warmup_samples=5)
        rng = np.random.default_rng(42)

        report = None
        for _ in range(20):
            rates = rng.normal(0.1, 0.005, size=10)
            rates = np.clip(rates, 0, 1)
            report = detector.update(rates)

        assert report is not None
        assert report.status in (DriftStatus.STABLE, DriftStatus.WARNING)

    def test_drift_detection(self):
        """A clear shift should trigger drift detection."""
        detector = EWMADriftDetector(warmup_samples=10, alarm_sigma=2.0)

        # Warmup with baseline
        for _ in range(15):
            rates = np.full(10, 0.05)
            detector.update(rates)

        # Apply a large shift
        report = None
        for _ in range(10):
            rates = np.full(10, 0.30)  # 6x increase
            report = detector.update(rates)

        assert report is not None
        assert report.status in (DriftStatus.DRIFT_DETECTED, DriftStatus.SEVERE)
        assert report.magnitude > 0


class TestCUSUMDriftDetector:
    """Test CUSUM drift detection."""

    def test_stable_no_alarm(self):
        detector = CUSUMDriftDetector(warmup_samples=10)

        report = None
        for _ in range(30):
            rates = np.full(5, 0.1)
            report = detector.update(rates)

        assert report is not None
        assert report.status == DriftStatus.STABLE

    def test_sudden_shift_alarm(self):
        detector = CUSUMDriftDetector(warmup_samples=10, threshold=3.0)

        # Warmup
        for _ in range(15):
            rates = np.full(5, 0.05)
            detector.update(rates)

        # Sudden shift
        report = None
        for _ in range(10):
            rates = np.full(5, 0.25)
            report = detector.update(rates)

        assert report is not None
        assert report.status == DriftStatus.DRIFT_DETECTED


class TestCompositeDriftDetector:
    """Test composite drift detection."""

    def test_composite_reports_worst(self):
        detector = CompositeDriftDetector(warmup_samples=10)

        # Warmup
        for _ in range(15):
            rates = np.full(5, 0.05)
            detector.update(rates)

        # Apply drift
        report = None
        for _ in range(10):
            rates = np.full(5, 0.3)
            report = detector.update(rates)

        # Should catch drift via at least one method
        assert report is not None
        assert report.status != DriftStatus.STABLE or report.magnitude > 0
