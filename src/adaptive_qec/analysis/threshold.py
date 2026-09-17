"""
Threshold scaling analysis for QEC experiments.

Computes the Lambda ratio (Λ) — the key metric that determines whether
a QEC system is operating below the fault-tolerant threshold.

    Λ = p_L(d) / p_L(d+2)

If Λ > 1, the system is below threshold: increasing code distance
exponentially suppresses logical errors.

If Λ < 1, the system is above threshold: the physical error rate is
too high for the code to help.

Reference values:
    - Google Willow (2025): Λ ≈ 2.14 ± 0.02
    - Theoretical surface code threshold: p_th ≈ 1%

This module also fits the phenomenological threshold model:
    p_L = A · (p / p_th)^((d+1)/2)

to extract the effective threshold p_th and scaling constant A.

Sources:
    - Google Quantum AI, "Quantum error correction below the surface code
      threshold", Nature (2025), arXiv:2408.13687
    - Fowler et al., "Surface codes: Towards practical large-scale quantum
      computation", Phys. Rev. A 86, 032324 (2012)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy import optimize

from adaptive_qec.decoders.base import DecoderMetrics

logger = logging.getLogger(__name__)


@dataclass
class ThresholdFit:
    """Result of fitting the threshold model to experimental data."""

    # Fitted parameters
    p_threshold: float  # estimated threshold error rate
    A: float  # scaling prefactor
    fit_residual: float  # sum of squared residuals

    # Input data
    distances: list[int]
    logical_error_rates: list[float]
    physical_error_rate: float

    # Derived
    lambda_ratios: dict[str, float] = field(default_factory=dict)
    is_below_threshold: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "p_threshold": round(self.p_threshold, 6),
            "A": round(self.A, 6),
            "fit_residual": round(self.fit_residual, 8),
            "distances": self.distances,
            "logical_error_rates": [round(r, 6) for r in self.logical_error_rates],
            "physical_error_rate": round(self.physical_error_rate, 6),
            "lambda_ratios": {
                k: round(v, 4) for k, v in self.lambda_ratios.items()
            },
            "is_below_threshold": self.is_below_threshold,
        }


class ThresholdAnalyzer:
    """
    Computes threshold scaling metrics from distance-sweep experiments.

    Usage:
        analyzer = ThresholdAnalyzer()

        # From individual metrics
        analyzer.add_result(distance=3, metrics=d3_metrics, p_phys=0.003)
        analyzer.add_result(distance=5, metrics=d5_metrics, p_phys=0.003)
        analyzer.add_result(distance=7, metrics=d7_metrics, p_phys=0.003)

        # Analyze
        fit = analyzer.fit_threshold_model()
        lambda_val = analyzer.compute_lambda(d_low=3, d_high=5)
    """

    def __init__(self) -> None:
        self._results: dict[int, dict[str, Any]] = {}

    def add_result(
        self,
        distance: int,
        metrics: DecoderMetrics,
        physical_error_rate: float,
    ) -> None:
        """
        Add an experimental result for a given code distance.

        Args:
            distance: code distance (must be odd: 3, 5, 7, ...)
            metrics: decoder metrics from the experiment
            physical_error_rate: the physical error rate used
        """
