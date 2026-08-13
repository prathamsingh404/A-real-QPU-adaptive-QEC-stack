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
