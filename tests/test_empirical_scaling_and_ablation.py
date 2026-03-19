"""
Tests for Phase 10: Empirical Scaling & Ablation Analysis.
"""

import pytest

from adaptive_qec.evaluation.empirical_scaling_and_ablation import (
    AdaptationPhaseDiagram,
    AblationRow,
    ComponentAblationStudy,
    ComprehensiveMetrics,
    ParetoFrontierAnalyzer,
    ParetoPoint,
    benchmark_decision_accuracy,
    compute_calibration_curve,
    compute_distance_scaling,
)


def test_distance_scaling():
    """Verify code distance scaling d in {3, 5} and Lambda calculation."""
    res = compute_distance_scaling(distances=(3, 5))
    assert res.distances == [3, 5]
    assert "3->5" in res.lambda_static
    assert res.lambda_static["3->5"] > 0.0


def test_comprehensive_metrics_structure():
    """Verify 5-domain comprehensive metric table."""
    m = ComprehensiveMetrics(
        ler_static=0.012,
        ler_adaptive=0.010,
        delta_ler_pct=-16.6,
        lambda_factor=1.6,
        gross_value_v_gross=0.002,
        net_value_v_net=0.002,
        efficiency_eta=0.85,
        switches_per_k_shots=1.5,
        rollback_rate_pct=0.0,
        abstention_rate_pct=85.0,
        syndrome_pred_rmse=0.008,
        detection_latency_rounds=4.1,
        false_alarm_rate_pct=1.8,
        decode_latency_us=45.0,
        decision_latency_us=12.0,
        throughput_shots_per_sec=18500.0,
        drift_tracking_r2=0.92,
        dd_crossover_margin_us=18.4,
    )
    d = m.to_dict()
    assert "QEC" in d
    assert "Adaptation" in d
    assert "Estimation" in d
    assert "Systems" in d
    assert "Hardware" in d
    assert d["QEC"]["LER_static"] == 0.012
