"""
Statistical significance testing for adaptive-vs-static comparisons.

Implements hypothesis tests to rigorously determine whether the
adaptive controller's improvement over the static baseline is
statistically significant, not just due to random variation.

Tests:
    1. Two-sample t-test (Welch's) — error rate comparison.
    2. Mann-Whitney U test — non-parametric alternative.
    3. Bootstrap confidence interval — difference in means.
    4. Benjamini-Hochberg correction — multiple comparisons.

All p-values and confidence intervals are reported with full
methodology documentation for paper reproducibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class HypothesisTestResult:
    """Result of a single hypothesis test."""
    test_name: str
    null_hypothesis: str
    alternative_hypothesis: str
    test_statistic: float
    p_value: float
    significant: bool
    alpha: float
    effect_size: float
    confidence_interval: tuple[float, float]
    sample_sizes: tuple[int, int]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "null_hypothesis": self.null_hypothesis,
            "alternative_hypothesis": self.alternative_hypothesis,
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "significant": self.significant,
            "alpha": self.alpha,
            "effect_size": self.effect_size,
            "confidence_interval": list(self.confidence_interval),
            "sample_sizes": list(self.sample_sizes),
            "notes": self.notes,
        }


@dataclass
class ComparisonReport:
    """Complete statistical comparison between two controllers."""
    baseline_name: str
    treatment_name: str
    baseline_mean: float
    treatment_mean: float
    absolute_improvement: float
    relative_improvement: float
    tests: list[HypothesisTestResult]
    bootstrap_ci: tuple[float, float]
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_name": self.baseline_name,
