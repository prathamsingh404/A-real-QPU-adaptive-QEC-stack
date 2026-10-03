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

        assert isinstance(analysis, LeakageAnalysis)
        assert analysis.total_rounds == 30
        assert analysis.total_detectors == 8
        assert len(analysis.leaked_qubits) == 0
        assert analysis.estimated_leakage_rate == 0.0

    def test_persistent_leakage_detected(self):
        """A qubit stuck firing consecutively should be flagged as leaked."""
        rng = np.random.default_rng(42)
        trace = (rng.random((40, 10)) < 0.02).astype(np.uint8)

        # Inject persistent leakage on detector 4 from round 10 to 35 (25 consecutive rounds)
        trace[10:35, 4] = 1

        detector = LeakageDetector(
            min_persistence=5,
            autocorr_threshold=0.25,
            firing_rate_threshold=0.3,
        )
        analysis = detector.analyze(trace)

        assert len(analysis.leaked_qubits) >= 1
        leaked_indices = [q.detector_index for q in analysis.leaked_qubits]
        assert 4 in leaked_indices

        q4 = next(q for q in analysis.leaked_qubits if q.detector_index == 4)
        assert q4.onset_round == 10
        assert q4.persistence_length >= 25
        assert q4.autocorrelation > 0.3
        assert q4.confidence > 0.5

    def test_longest_streak_utility(self):
        arr = np.array([0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0])
        streak, onset = LeakageDetector._longest_streak(arr)
        assert streak == 4
        assert onset == 8


class TestLeakageRateEstimator:
    """Tests for LeakageRateEstimator."""

    def test_rate_estimation(self):
        estimator = LeakageRateEstimator()

        # Mock analysis with one leaked qubit persisting 10 rounds
        mock_analysis = LeakageAnalysis(
            total_rounds=100,
            total_detectors=10,
            leaked_qubits=[
                LeakedQubit(
                    detector_index=2,
                    onset_round=15,
                    persistence_length=10,
                    autocorrelation=0.8,
                    firing_rate=0.7,
                    confidence=0.85,
                )
            ],
            estimated_leakage_rate=0.001,
        )

        estimator.add_analysis(mock_analysis)
        rates = estimator.estimate_rates()

        assert "gamma_leakage" in rates
        assert "gamma_seepage" in rates
        assert "p_leak_steady" in rates
        assert rates["gamma_leakage"] > 0
        assert pytest.approx(rates["gamma_seepage"], rel=1e-2) == 0.1  # 1 / 10 rounds
        assert rates["p_leak_steady"] > 0


class TestDigitalTwinLeakage:
    """Tests for Digital Twin integration with leakage."""

    def test_update_leakage_from_analysis(self):
        twin = HardwareDigitalTwin(num_qubits=10)

        analysis = LeakageAnalysis(
            total_rounds=50,
            total_detectors=10,
            leaked_qubits=[
                LeakedQubit(
                    detector_index=3,
                    onset_round=5,
                    persistence_length=20,
                    autocorrelation=0.75,
                    firing_rate=0.8,
                    confidence=0.9,
                )
            ],
        )

        twin.update_leakage_from_analysis(analysis)
        q3 = twin.get_qubit_state(3)
        assert q3 is not None
        assert q3.leakage_probability == 0.9

        q1 = twin.get_qubit_state(1)
        assert q1 is not None
        assert q1.leakage_probability == 0.0
