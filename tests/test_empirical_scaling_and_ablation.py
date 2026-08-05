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


def test_pareto_frontier_analysis():
    """Verify non-dominated Pareto front calculation."""
    points = [
        ParetoPoint("Static_MWPM", ler=0.012, latency_us=10.0, switches_per_k_shots=0.0),
        ParetoPoint("Naive_Adaptive", ler=0.029, latency_us=80.0, switches_per_k_shots=40.0), # Dominated
        ParetoPoint("CG_CCADS", ler=0.011, latency_us=25.0, switches_per_k_shots=2.4),
        ParetoPoint("Heavy_Neural", ler=0.009, latency_us=120.0, switches_per_k_shots=5.0),
    ]

    front = ParetoFrontierAnalyzer.identify_pareto_front(points, objective_x="latency_us")
    names = [p.config_name for p in front]
    assert "Static_MWPM" in names
    assert "CG_CCADS" in names
    assert "Heavy_Neural" in names
    assert "Naive_Adaptive" not in names  # Strictly dominated


def test_adaptation_phase_diagram():
    """Verify 2D phase diagram classification into 3 regimes."""
    cells = AdaptationPhaseDiagram.construct_diagram(
        drift_rates=(0.001, 1.0, 10.0),
        amplitudes=(0.002, 0.05),
    )
    regimes = {c.optimal_regime for c in cells}
    assert "THRASHING_ZONE" in regimes
    assert "STAY_OPTIMAL" in regimes
    assert "ADAPT_OPTIMAL" in regimes


def test_decision_accuracy_benchmark():
    """Verify controller decision precision, recall, and false adaptation rate."""
    decisions = ["ADAPT", "ADAPT", "STAY", "STAY", "ADAPT"]
    ground_truth = [True, True, False, False, False] # 2 TP, 2 TN, 1 FP
