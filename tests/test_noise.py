"""Tests for noise statistics and drift detection."""

import numpy as np
import pytest

from adaptive_qec.noise.statistics import (
    compute_detector_statistics,
    compute_temporal_correlation,
    compute_spatial_correlation,
)
from adaptive_qec.noise.characterization import (
    TLSDefectState,
    TLSEnsemble,
    TLSFluctuator,
    TLSTelegraphNoise,
)
from adaptive_qec.noise.drift import (
    CompositeDriftDetector,
    CUSUMDriftDetector,
    DriftStatus,
    EWMADriftDetector,
    OrnsteinUhlenbeckDrift,
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


class TestOrnsteinUhlenbeckAndHeron:
    """Test Ornstein-Uhlenbeck drift trajectory and IBM Heron noise generator."""

    def test_ou_drift_trajectory_bounds(self):
        from adaptive_qec.engine.noise_models import DriftTrajectory
        ou = DriftTrajectory(theta=0.1, mu=0.005, sigma=0.001, min_val=0.001, max_val=0.02)
        path = ou.sample_path(steps=100, seed=42)
        assert len(path) == 100
        assert np.all(path >= 0.001)
        assert np.all(path <= 0.02)
        assert not np.isnan(path).any()

    def test_burst_noise_model_injection(self):
        from adaptive_qec.engine.noise_models import BurstNoiseModel
        model = BurstNoiseModel(burst_probability_per_step=0.1, peak_error_multiplier=5.0)
        base = np.full(50, 0.005)
        perturbed, bursts = model.inject_bursts(base, seed=123)
        assert len(perturbed) == 50
        assert len(bursts) > 0
        assert np.max(perturbed) > 0.005

    def test_ibm_heron_noise_trajectory_generation(self):
        from adaptive_qec.engine.noise_models import generate_ibm_heron_noise_trajectory
        res = generate_ibm_heron_noise_trajectory(steps=80, include_bursts=True, seed=999)
        assert "cz_error" in res
        assert "readout_error" in res
        assert "t1_us" in res
        assert "t2_us" in res
        assert len(res["cz_error"]) == 80
        assert res["reference_qv"] == 512
        assert "Heron" in res["device_architecture"]


class TestOrnsteinUhlenbeckDrift:
    """Test Ornstein-Uhlenbeck continuous-time drift process."""

    def test_parameter_validation(self):
        # theta must be > 0
        with pytest.raises(ValueError, match="theta"):
            OrnsteinUhlenbeckDrift(theta=0.0)
        with pytest.raises(ValueError, match="theta"):
            OrnsteinUhlenbeckDrift(theta=-0.5)

        # sigma must be >= 0
        with pytest.raises(ValueError, match="sigma"):
            OrnsteinUhlenbeckDrift(sigma=-0.01)

        # dt must be > 0
        with pytest.raises(ValueError, match="dt"):
            OrnsteinUhlenbeckDrift(dt=0.0)
        with pytest.raises(ValueError, match="dt"):
            OrnsteinUhlenbeckDrift(dt=-1.0)

        # num_steps must be > 0
        ou = OrnsteinUhlenbeckDrift(theta=0.5, mu=0.02, sigma=0.01)
        with pytest.raises(ValueError, match="num_steps"):
            ou.generate_trajectory(num_steps=0)

    def test_generate_trajectory_shape_and_initial(self):
        ou = OrnsteinUhlenbeckDrift(theta=0.2, mu=0.03, sigma=0.005, dt=0.5, seed=42)
        traj = ou.generate_trajectory(num_steps=100, initial_value=0.08)
        assert isinstance(traj, np.ndarray)
        assert traj.shape == (100,)
        assert np.isclose(traj[0], 0.08)

        # single step trajectory
        traj_single = ou.generate_trajectory(num_steps=1, initial_value=0.05)
        assert traj_single.shape == (1,)
        assert np.isclose(traj_single[0], 0.05)

    def test_mean_reversion(self):
        # Start far above mu, high theta, low sigma => should revert towards mu
        ou = OrnsteinUhlenbeckDrift(theta=1.5, mu=0.02, sigma=0.001, dt=0.1, seed=42)
        traj = ou.generate_trajectory(num_steps=200, initial_value=0.20)
        # Final values should be close to mu (0.02)
        assert np.abs(traj[-1] - 0.02) < 0.01
        assert np.mean(traj[-50:]) < 0.05

    def test_zero_volatility_deterministic_decay(self):
        # When sigma = 0, deterministic exponential decay towards mu
        theta = 0.5
        dt = 0.1
        mu = 0.01
        x0 = 0.10
        ou = OrnsteinUhlenbeckDrift(theta=theta, mu=mu, sigma=0.0, dt=dt)
        traj = ou.generate_trajectory(num_steps=50, initial_value=x0)
        for t in range(49):
            expected = traj[t] + theta * (mu - traj[t]) * dt
            assert np.isclose(traj[t + 1], expected)

    def test_step_and_reset(self):
        ou = OrnsteinUhlenbeckDrift(theta=0.5, mu=0.05, sigma=0.002, seed=123)
        ou.reset(initial_value=0.10)
        assert np.isclose(ou.current_value, 0.10)
        next_val = ou.step()
        assert ou.current_value == next_val


class TestTLSFluctuator:
    """Test Two-Level System (TLS) defect telegraph fluctuator model."""

    def test_parameter_validation(self):
        with pytest.raises(ValueError, match="Switching rates"):
            TLSFluctuator(rate_01=-0.1)
        with pytest.raises(ValueError, match="Decoherence rates"):
            TLSFluctuator(gamma_0=-1.0)
        with pytest.raises(ValueError, match="dt"):
            TLSFluctuator(dt=0.0)
        with pytest.raises(ValueError, match="initial_state"):
            TLSFluctuator(initial_state=2)

    def test_steady_state_and_properties(self):
        tls = TLSFluctuator(
            rate_01=0.02,
            rate_10=0.08,
            gamma_0=1e4,
            gamma_1=5e4,
            dt=1.0,
        )
        # p0 = 0.08 / 0.10 = 0.80, p1 = 0.02 / 0.10 = 0.20
        assert np.isclose(tls.p_state_0_steady, 0.80)
        assert np.isclose(tls.p_state_1_steady, 0.20)
        assert np.isclose(tls.correlation_time, 10.0)
        expected_mean = 0.8 * 1e4 + 0.2 * 5e4
        assert np.isclose(tls.effective_mean_rate, expected_mean)

    def test_trajectory_generation(self):
        tls = TLSFluctuator(
            rate_01=0.1,
            rate_10=0.1,
            gamma_0=1e4,
            gamma_1=3e4,
            dt=0.5,
            seed=42,
        )
        states, rates = tls.generate_trajectory(num_steps=500, initial_state=0)
        assert states.shape == (500,)
        assert rates.shape == (500,)
        assert set(np.unique(states)).issubset({0, 1})
        # Discrete rates match state
        assert np.all(rates[states == 0] == 1e4)
        assert np.all(rates[states == 1] == 3e4)
        # Both states should be visited over 500 steps
        assert 0 in states and 1 in states

    def test_zero_switching_rate_frozen(self):
        tls = TLSFluctuator(rate_01=0.0, rate_10=0.0, gamma_0=1e4, gamma_1=5e4, initial_state=1)
        states, rates = tls.generate_trajectory(num_steps=50)
        assert np.all(states == 1)
        assert np.all(rates == 5e4)

    def test_telegraph_noise_convenience(self):
        tls = TLSTelegraphNoise(rate_01=0.05, rate_10=0.05, gamma_0=10.0, gamma_1=20.0, seed=10)
        noise = tls.generate_telegraph_noise(num_steps=100)
        assert noise.shape == (100,)
        assert np.all(np.isin(noise, [10.0, 20.0]))

    def test_tls_ensemble(self):
        ensemble = TLSEnsemble.create_dutta_horn_ensemble(
            num_fluctuators=5,
            min_rate=0.01,
            max_rate=0.5,
            gamma_defect=1e3,
            base_rate=1e4,
            seed=42,
        )
        traj = ensemble.generate_trajectory(num_steps=100)
        assert traj.shape == (100,)
        # All rates >= base_rate
        assert np.all(traj >= 1e4)
        # Some variation over time
        assert np.std(traj) > 0.0


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

