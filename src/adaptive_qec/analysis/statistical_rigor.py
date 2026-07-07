"""
Statistical Rigor & Pre-Registration Infrastructure for Adaptive QEC.

In non-stationary QEC environments, shots within a single run are temporally
correlated due to drift and memory effects. Reporting p-values computed from
pooled shots commits pseudo-replication.

This module enforces:
1. Independent noise trajectory as the primary statistical unit (N >= 20, target 30).
2. Strict seed partitioning:
   - Train seeds: 0–9 (exploratory policy tuning)
   - Validation seeds: 10–19 (hyperparameter selection, conformal calibration)
   - Test seeds: 20–49 (held-out confirmatory evaluation, 30 independent trajectories)
3. Trajectory-paired delta analysis: Δ_i = LER_{adaptive, i} - LER_{static, i}.
4. Formal statistical power analysis (ensuring >= 80% power at α = 0.05).
5. Pre-registered confirmatory vs. exploratory regime boundaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class SeedSplit(str, Enum):
    """Seed partitions for strict out-of-sample statistical integrity."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class SeedPartition:
    """Strict seed partitioning ensuring no data leakage into confirmatory tests."""
    TRAIN_SEEDS: List[int] = list(range(0, 100))        # 100 tuning seeds: 0–99
    VAL_SEEDS: List[int] = list(range(100, 120))        # 20 validation seeds: 100–119
    TEST_SEEDS: List[int] = list(range(1000, 1030))     # 30 held-out confirmatory seeds: 1000–1029

    @classmethod
    def get_seeds(cls, split: str | SeedSplit) -> List[int]:
        split_str = str(split).lower()
        if "train" in split_str:
            return list(cls.TRAIN_SEEDS)
        elif "val" in split_str:
            return list(cls.VAL_SEEDS)
        elif "test" in split_str:
            return list(cls.TEST_SEEDS)
        else:
            raise ValueError(f"Unknown split: {split}. Choose 'train', 'validation', or 'test'.")

    @classmethod
    def validate_no_overlap(cls) -> bool:
        s_train = set(cls.TRAIN_SEEDS)
        s_val = set(cls.VAL_SEEDS)
        s_test = set(cls.TEST_SEEDS)
        assert len(s_train.intersection(s_val)) == 0, "Train and Validation seeds overlap!"
        assert len(s_train.intersection(s_test)) == 0, "Train and Test seeds overlap!"
        assert len(s_val.intersection(s_test)) == 0, "Validation and Test seeds overlap!"
        return True


@dataclass(frozen=True)
class TrajectoryResult:
    """Statistical unit: one independent noise trajectory realization."""
    trajectory_id: str
    scenario_name: str
    seed: int
    split: str
    static_ler: float
    adaptive_ler: float
    delta_ler: float  # adaptive_ler - static_ler (negative is better for adaptive)
    adaptation_count: int
    stay_count: int
    rollback_count: int
    total_shots: int
    adaptation_cost: float = 0.0
    net_value: float = 0.0
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairedTrajectorySummary:
    """Results of a paired trajectory statistical evaluation across N realizations."""
    scenario_name: str
    num_trajectories: int
    mean_static_ler: float
    mean_adaptive_ler: float
    mean_delta_ler: float             # delta = adaptive - static
    std_delta_ler: float
    sem_delta_ler: float              # Standard error of mean delta
    ci_95_t: Tuple[float, float]       # Student's t 95% CI
    ci_95_bootstrap: Tuple[float, float]
    t_statistic: float
    t_p_value: float
    wilcoxon_stat: Optional[float]
    wilcoxon_p_value: Optional[float]
    cohens_d: float                   # Effect size d_z = mean / std
    statistically_significant: bool
    direction: str                    # "ADAPTIVE_SUPERIOR", "STATIC_SUPERIOR", "INCONCLUSIVE"
    raw_deltas: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "num_trajectories": self.num_trajectories,
            "mean_static_ler": self.mean_static_ler,
            "mean_adaptive_ler": self.mean_adaptive_ler,
            "mean_delta_ler": self.mean_delta_ler,
            "std_delta_ler": self.std_delta_ler,
            "sem_delta_ler": self.sem_delta_ler,
            "ci_95_t": list(self.ci_95_t),
            "ci_95_bootstrap": list(self.ci_95_bootstrap),
            "t_statistic": self.t_statistic,
            "t_p_value": self.t_p_value,
            "wilcoxon_stat": self.wilcoxon_stat,
            "wilcoxon_p_value": self.wilcoxon_p_value,
            "cohens_d": self.cohens_d,
            "statistically_significant": self.statistically_significant,
            "direction": self.direction,
        }


class PairedTrajectoryAnalysis:
    """Enforces rigorous paired comparison across independent noise trajectories."""

    @staticmethod
    def analyze_trajectories(
        trajectories: Sequence[TrajectoryResult],
        confidence: float = 0.95,
        n_bootstrap: int = 10000,
        seed: int = 42,
    ) -> PairedTrajectorySummary:
        """
        Analyze paired trajectory delta LER: Delta_i = LER_{adaptive, i} - LER_{static, i}.
        Negative Delta means adaptive achieved lower logical error rate.
        """
        if len(trajectories) < 2:
            raise ValueError(f"Need at least 2 trajectories for paired analysis, got {len(trajectories)}")

        scenario_name = trajectories[0].scenario_name
        deltas = np.array([t.delta_ler for t in trajectories], dtype=np.float64)
        static_lers = np.array([t.static_ler for t in trajectories], dtype=np.float64)
        adaptive_lers = np.array([t.adaptive_ler for t in trajectories], dtype=np.float64)

        n = len(deltas)
        mean_delta = float(np.mean(deltas))
        std_delta = float(np.std(deltas, ddof=1)) if n > 1 else 0.0
        sem_delta = std_delta / np.sqrt(n) if n > 0 else 0.0

        # Student's t distribution CI
        alpha = 1.0 - confidence
        t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
        ci_t = (mean_delta - t_crit * sem_delta, mean_delta + t_crit * sem_delta)

        # Non-parametric Bootstrap CI
        rng = np.random.default_rng(seed)
        boot_means = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            boot_means[b] = np.mean(deltas[idx])
        ci_boot = (
            float(np.percentile(boot_means, 100.0 * (alpha / 2.0))),
            float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0))),
        )

        # Paired t-test
        if std_delta > 1e-15:
            t_stat = mean_delta / sem_delta
            t_pval = float(2.0 * stats.t.sf(np.abs(t_stat), df=n - 1))
        else:
            t_stat = 0.0
            t_pval = 1.0

        # Wilcoxon signed-rank test (handles non-normal distribution)
        wilcoxon_stat: Optional[float] = None
        wilcoxon_pval: Optional[float] = None
        try:
            # Check if all deltas are identical
            if not np.allclose(deltas, 0.0) and np.any(np.abs(deltas - np.median(deltas)) > 1e-12):
                w_res = stats.wilcoxon(deltas, alternative="two-sided")
                wilcoxon_stat = float(w_res.statistic)
                wilcoxon_pval = float(w_res.pvalue)
        except Exception as e:
            logger.debug("Wilcoxon test skipped or failed: %s", e)

        # Cohen's d_z for paired samples
        cohens_d = mean_delta / std_delta if std_delta > 1e-15 else 0.0

        # Significance and direction
        significant = (t_pval < alpha) or (wilcoxon_pval is not None and wilcoxon_pval < alpha)
        if significant and mean_delta < 0:
            direction = "ADAPTIVE_SUPERIOR"
        elif significant and mean_delta > 0:
            direction = "STATIC_SUPERIOR"
        else:
            direction = "INCONCLUSIVE"
