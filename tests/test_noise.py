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


class TestOUProcess:
    """Tests for Ornstein-Uhlenbeck (OU) mean-reverting drift dynamics."""

    @staticmethod
    def simulate_ou(
        theta: float = 0.1,
        mu: float = 0.02,
        sigma: float = 0.005,
        dt: float = 1.0,
        n_steps: int = 500,
        x0: float = 0.08,
        seed: int = 42,
    ) -> np.ndarray:
        """Simulate Ornstein-Uhlenbeck process dX_t = theta*(mu - X_t)*dt + sigma*dW_t."""
        rng = np.random.default_rng(seed)
        x = np.zeros(n_steps)
        x[0] = x0
        for t in range(1, n_steps):
            dx = theta * (mu - x[t - 1]) * dt + sigma * np.sqrt(dt) * rng.normal()
            x[t] = x[t - 1] + dx
        return x

    def test_ou_mean_reversion(self):
        """OU process starting away from mu mean-reverts toward mu."""
        mu = 0.02
        traj = self.simulate_ou(theta=0.15, mu=mu, x0=0.08, n_steps=600)
        # Early values right after x0 = 0.08 are elevated
        assert np.mean(traj[:5]) > 0.05
        # Late values fluctuate around mu = 0.02
        assert abs(np.mean(traj[-200:]) - mu) < 0.004

    def test_ou_stationary_variance_bound(self):
        """Stationary variance of OU process converges near sigma^2 / (2 * theta)."""
        theta = 0.2
        sigma = 0.004
        mu = 0.015
        expected_var = (sigma ** 2) / (2.0 * theta)

        trajs = [self.simulate_ou(theta=theta, mu=mu, sigma=sigma, n_steps=1000, seed=s) for s in range(30)]
        late_points = [tr[-1] for tr in trajs]
        empirical_var = float(np.var(late_points))

        # Empirical variance is close to theoretical stationary variance
        assert abs(empirical_var - expected_var) / expected_var < 0.40


class TestBurstDetectorIntegration:
    """Tests integrating BurstDetector with noisy syndrome streams."""

    def test_burst_detection_on_spikes(self):
        """BurstDetector flags severe multi-detector defect spikes."""
        from adaptive_qec.noise.burst_detector import BurstDetector

        detector = BurstDetector(window_size=5, significance=0.001, min_defects_for_burst=15)
        rng = np.random.default_rng(42)

        # Baseline quiet syndromes (rounds x detectors)
        quiet = rng.binomial(1, 0.01, size=(50, 16)).astype(np.uint8)
        bursts_quiet = detector.analyze(quiet, num_detectors_per_round=16)
        assert len(bursts_quiet.bursts_detected) == 0

        # Inject massive cosmic-ray burst across all detectors
        spike = np.ones((5, 16), dtype=np.uint8)
        stream = np.vstack([quiet, spike, quiet])
        bursts_loud = detector.analyze(stream, num_detectors_per_round=16)
        assert len(bursts_loud.bursts_detected) > 0

