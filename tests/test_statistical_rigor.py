"""
Tests for Phase 8: Statistical Rigor & Power Analysis Infrastructure.
"""

import numpy as np
import pytest

from adaptive_qec.analysis.statistical_rigor import (
    ConfirmatoryProtocol,
    PairedTrajectoryAnalysis,
    PairedTrajectorySummary,
    PowerAnalysis,
    SeedPartition,
    SeedSplit,
    TrajectoryResult,
)


def test_seed_partitions_no_overlap():
    """Verify strict isolation of train (0-9), val (10-19), and test (20-49) seeds."""
    assert SeedPartition.validate_no_overlap()

    train_seeds = SeedPartition.get_seeds(SeedSplit.TRAIN)
    val_seeds = SeedPartition.get_seeds(SeedSplit.VALIDATION)
    test_seeds = SeedPartition.get_seeds(SeedSplit.TEST)

    assert len(train_seeds) == 100
    assert len(val_seeds) == 20
    assert len(test_seeds) == 30  # Confirmatory N=30

    assert set(train_seeds).isdisjoint(set(val_seeds))
    assert set(train_seeds).isdisjoint(set(test_seeds))
    assert set(val_seeds).isdisjoint(set(test_seeds))

    assert min(test_seeds) == 1000
    assert max(test_seeds) == 1029


def test_paired_trajectory_analysis_superiority():
    """Test paired trajectory statistics when adaptive strictly outperforms static."""
    trajectories = []
    rng = np.random.default_rng(12345)

    # 30 paired trajectories where adaptive is consistently better by ~0.015 LER
    for i in range(30):
        static_ler = float(rng.normal(0.18, 0.01))
        # Adaptive is better by ~0.015 with small variation
        adaptive_ler = static_ler - float(rng.normal(0.015, 0.003))
        delta = adaptive_ler - static_ler

        trajectories.append(
            TrajectoryResult(
                trajectory_id=f"traj_{i}",
                scenario_name="slow_drift",
                seed=20 + i,
                split="test",
                static_ler=static_ler,
                adaptive_ler=adaptive_ler,
                delta_ler=delta,
                adaptation_count=2,
                stay_count=18,
                rollback_count=0,
                total_shots=1000,
            )
        )

    summary = PairedTrajectoryAnalysis.analyze_trajectories(trajectories)

    assert summary.num_trajectories == 30
    assert summary.mean_delta_ler < -0.010
    assert summary.statistically_significant is True
    assert summary.direction == "ADAPTIVE_SUPERIOR"
    assert summary.ci_95_t[1] < 0.0  # Entire 95% CI is strictly below zero
    assert summary.ci_95_bootstrap[1] < 0.0
    assert summary.t_p_value < 0.001
    assert summary.cohens_d < -2.0


def test_paired_trajectory_analysis_inconclusive():
    """Test paired trajectory statistics when adaptation has no effect (null hypothesis)."""
    trajectories = []
    rng = np.random.default_rng(999)

    for i in range(30):
        static_ler = 0.170
        # Zero true difference plus symmetric noise
        adaptive_ler = static_ler + float(rng.normal(0.0, 0.002))
        delta = adaptive_ler - static_ler

        trajectories.append(
            TrajectoryResult(
                trajectory_id=f"traj_null_{i}",
                scenario_name="stationary_baseline",
                seed=20 + i,
                split="test",
                static_ler=static_ler,
                adaptive_ler=adaptive_ler,
                delta_ler=delta,
                adaptation_count=0,
                stay_count=20,
                rollback_count=0,
                total_shots=1000,
            )
        )

    summary = PairedTrajectoryAnalysis.analyze_trajectories(trajectories)

    assert summary.num_trajectories == 30
    assert summary.direction == "INCONCLUSIVE"
    # CI should span zero
    assert summary.ci_95_t[0] < 0.0 < summary.ci_95_t[1]


def test_power_analysis_monotonicity():
    """Verify power curves increase monotonically with effect size and sample size."""
    # Power increases with effect size
    p1 = PowerAnalysis.compute_power(effect_size_d=0.3, n_samples=30)
    p2 = PowerAnalysis.compute_power(effect_size_d=0.6, n_samples=30)
    p3 = PowerAnalysis.compute_power(effect_size_d=0.9, n_samples=30)
    assert p1 < p2 < p3

    # Power increases with sample size
    p_n10 = PowerAnalysis.compute_power(effect_size_d=0.6, n_samples=10)
    p_n30 = PowerAnalysis.compute_power(effect_size_d=0.6, n_samples=30)
    p_n50 = PowerAnalysis.compute_power(effect_size_d=0.6, n_samples=50)
    assert p_n10 < p_n30 < p_n50

    # N=30 achieves >= 80% power for d >= 0.55
    assert PowerAnalysis.compute_power(effect_size_d=0.55, n_samples=30) >= 0.80

    # Sample size calculator
    req_n = PowerAnalysis.sample_size_for_power(effect_size_d=0.6, target_power=0.80)
    assert req_n <= 30


def test_confirmatory_protocol_regimes():
    """Verify confirmatory vs exploratory classification."""
    assert ConfirmatoryProtocol.is_confirmatory("stationary_baseline")
    assert ConfirmatoryProtocol.is_confirmatory("slow_drift")
    assert ConfirmatoryProtocol.is_confirmatory("obvious_adaptation")

    assert not ConfirmatoryProtocol.is_confirmatory("high_freq_jitter")
    assert not ConfirmatoryProtocol.is_confirmatory("switching_trap")
    assert not ConfirmatoryProtocol.is_confirmatory("recovery")
