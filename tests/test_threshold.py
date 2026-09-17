"""Tests for Threshold Scaling Analysis (Problem 5).

Validates Lambda ratio computation (Λ), phenomenological threshold fitting,
and automated distance sweep experiments.
"""

import numpy as np
import pytest

from adaptive_qec.analysis.threshold import ThresholdAnalyzer, ThresholdFit
from adaptive_qec.config import NoiseConfig
from adaptive_qec.decoders.base import DecoderMetrics
from adaptive_qec.experiment.distance_sweep import DistanceSweep, DistanceSweepResults


def _mock_metrics(shots: int, errors: int, latency_us: float = 5.0) -> DecoderMetrics:
    """Helper to create realistic DecoderMetrics for testing."""
    ler = errors / shots if shots > 0 else 0.0
    return DecoderMetrics(
        total_shots=shots,
        num_logical_errors=errors,
        logical_error_rate=ler,
        decode_time_s=0.01,
        per_shot_latency_us=np.full(shots, latency_us),
        latency_mean_us=latency_us,
        latency_p50_us=latency_us,
        latency_p95_us=latency_us,
        latency_p99_us=latency_us,
        latency_p999_us=latency_us,
        throughput_shots_per_s=10000.0,
        peak_memory_mb=1.5,
    )


class TestThresholdAnalyzer:
    """Tests for ThresholdAnalyzer class."""

    def test_even_distance_raises(self):
        analyzer = ThresholdAnalyzer()
        with pytest.raises(ValueError, match="Code distance must be odd"):
            analyzer.add_result(distance=4, metrics=_mock_metrics(100, 5), physical_error_rate=0.005)

    def test_add_and_compute_lambda_below_threshold(self):
        analyzer = ThresholdAnalyzer()
        # Below threshold: d=3 has higher logical error than d=5
        analyzer.add_result(3, _mock_metrics(10000, 200), physical_error_rate=0.003)
        analyzer.add_result(5, _mock_metrics(10000, 50), physical_error_rate=0.003)

        lam = analyzer.compute_lambda(3, 5)
        # p_L(3) = 0.02, p_L(5) = 0.005 -> Lambda = 4.0
        assert pytest.approx(lam, rel=1e-3) == 4.0
        assert lam > 1.0

    def test_add_and_compute_lambda_above_threshold(self):
        analyzer = ThresholdAnalyzer()
        # Above threshold: d=3 has lower logical error than d=5
        analyzer.add_result(3, _mock_metrics(10000, 50), physical_error_rate=0.03)
        analyzer.add_result(5, _mock_metrics(10000, 150), physical_error_rate=0.03)

        lam = analyzer.compute_lambda(3, 5)
        # p_L(3) = 0.005, p_L(5) = 0.015 -> Lambda = 0.333
        assert pytest.approx(lam, rel=1e-2) == 0.333
        assert lam < 1.0
