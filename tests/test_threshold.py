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

    def test_missing_distance_raises(self):
        analyzer = ThresholdAnalyzer()
        analyzer.add_result(3, _mock_metrics(100, 5), physical_error_rate=0.005)
        with pytest.raises(ValueError, match="Need results for both"):
            analyzer.compute_lambda(3, 7)

    def test_fit_threshold_model(self):
        analyzer = ThresholdAnalyzer()
        analyzer.add_result(3, _mock_metrics(10000, 100), physical_error_rate=0.003)
        analyzer.add_result(5, _mock_metrics(10000, 20), physical_error_rate=0.003)
        analyzer.add_result(7, _mock_metrics(10000, 4), physical_error_rate=0.003)

        fit = analyzer.fit_threshold_model()
        assert isinstance(fit, ThresholdFit)
        assert fit.is_below_threshold is True
        assert fit.p_threshold > 0.003  # Threshold should be higher than physical error rate
        assert fit.A > 0
        assert fit.r_squared > 0.95  # Exponential decay fits well
        assert fit.p_threshold_ci[0] <= fit.p_threshold <= fit.p_threshold_ci[1]
        assert "r_squared" in fit.to_dict()
        assert "p_threshold_ci" in fit.to_dict()
        assert "fit_method" in fit.to_dict()
        assert "d3_to_d5" in fit.lambda_ratios
        assert "d5_to_d7" in fit.lambda_ratios

    def test_find_threshold_crossing(self):
        # Physical error rates across threshold
        p_phys = [0.004, 0.008, 0.012, 0.016]
        # At p=0.004, 0.008 (below p_th ~ 0.010): d=5 has lower LER than d=3
        # At p=0.012, 0.016 (above p_th ~ 0.010): d=5 has higher LER than d=3
        ler_matrix = [
            [0.010, 0.018, 0.026, 0.035],  # d=3
            [0.004, 0.014, 0.032, 0.055],  # d=5
        ]
        p_cross = ThresholdAnalyzer.find_threshold_crossing(
            distances=[3, 5],
            physical_error_rates=p_phys,
            logical_error_matrix=ler_matrix,
        )
        assert p_cross is not None
        assert 0.008 < p_cross < 0.012

        # Non-crossing case (always below threshold)
        ler_below = [
            [0.020, 0.030],
            [0.005, 0.010],
        ]
        assert ThresholdAnalyzer.find_threshold_crossing([3, 5], [0.001, 0.002], ler_below) is None

    def test_scaling_table(self):
        analyzer = ThresholdAnalyzer()
        analyzer.add_result(3, _mock_metrics(1000, 20), physical_error_rate=0.003)
        analyzer.add_result(5, _mock_metrics(1000, 5), physical_error_rate=0.003)

        table = analyzer.scaling_table()
        assert len(table) == 2
        assert table[0]["distance"] == 3
        assert table[0]["lambda_vs_prev"] is None
        assert table[1]["distance"] == 5
        assert table[1]["lambda_vs_prev"] > 1.0


class TestDistanceSweep:
    """Tests for automated DistanceSweep runner."""

    def test_sweep_execution(self):
        noise = NoiseConfig()
        noise.gate.two_qubit = 0.008

        sweep = DistanceSweep(
            distances=[3],
            rounds_per_distance={3: 2},
            noise=noise,
            decoder_names=["mwpm", "union_find"],
            shots_per_distance=100,
            code_type="surface",
        )
        results = sweep.run()

        assert isinstance(results, DistanceSweepResults)
        assert len(results.results) == 2  # mwpm and union_find
        assert results.total_wall_time_s > 0

        mwpm_res = results.get_by_decoder("mwpm")
        assert len(mwpm_res) == 1
        assert mwpm_res[0].metrics.total_shots == 100
        assert 0 <= mwpm_res[0].metrics.logical_error_rate <= 1.0

        uf_res = results.get_by_decoder("union_find")
        assert len(uf_res) == 1
        assert uf_res[0].metrics.total_shots == 100
        assert 0 <= uf_res[0].metrics.logical_error_rate <= 1.0

        d_dict = results.to_dict()
        assert "results" in d_dict
        assert len(d_dict["results"]) == 2
