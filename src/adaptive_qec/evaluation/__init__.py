"""
Evaluation module initialization.
"""

from adaptive_qec.evaluation.counterfactual import (
    CounterfactualEngine,
    DisagreementMatrix,
    PairedEvaluationResult,
    wilson_score_interval,
)
from adaptive_qec.evaluation.baseline_ladder import (
    BaselineLadder,
    BaselineArm,
    BaselineRung,
    BaselineLevel,
    BaselineLevelInfo,
    BASELINE_LEVEL_REGISTRY,
    BaselineDeltaBenchmark,
    compute_baseline_deltas,
)

__all__ = [
    "CounterfactualEngine",
    "DisagreementMatrix",
    "PairedEvaluationResult",
    "wilson_score_interval",
    "BaselineLadder",
    "BaselineArm",
    "BaselineRung",
    "BaselineLevel",
    "BaselineLevelInfo",
    "BASELINE_LEVEL_REGISTRY",
    "BaselineDeltaBenchmark",
    "compute_baseline_deltas",
]
