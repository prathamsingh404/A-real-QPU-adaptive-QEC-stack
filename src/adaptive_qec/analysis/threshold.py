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
        if distance % 2 == 0:
            raise ValueError(f"Code distance must be odd, got {distance}")

        self._results[distance] = {
            "distance": distance,
            "logical_error_rate": metrics.logical_error_rate,
            "num_logical_errors": metrics.num_logical_errors,
            "total_shots": metrics.total_shots,
            "physical_error_rate": physical_error_rate,
            "decode_time_s": metrics.decode_time_s,
            "latency_mean_us": metrics.latency_mean_us,
        }

        # Compute Wilson score 95% CI
        n = metrics.total_shots
        p = metrics.logical_error_rate
        z = 1.96
        denom = 1.0 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        delta = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        self._results[distance]["ci_lower"] = max(0.0, float(center - delta))
        self._results[distance]["ci_upper"] = min(1.0, float(center + delta))

        logger.info(
            f"Added d={distance}: LER={metrics.logical_error_rate:.6f} "
            f"[{self._results[distance]['ci_lower']:.6f}, "
            f"{self._results[distance]['ci_upper']:.6f}] "
            f"({metrics.num_logical_errors}/{metrics.total_shots})"
        )

    def compute_lambda(
        self,
        d_low: int,
        d_high: int,
    ) -> float:
        """
        Compute Λ = p_L(d_low) / p_L(d_high).

        Λ > 1 indicates below-threshold operation.

        Args:
            d_low: lower code distance
            d_high: higher code distance (should be d_low + 2)

        Returns:
            Lambda ratio. Returns 0.0 if either error rate is zero.
        """
        if d_low not in self._results or d_high not in self._results:
            available = sorted(self._results.keys())
            raise ValueError(
                f"Need results for both d={d_low} and d={d_high}. "
                f"Available: {available}"
            )

        p_low = self._results[d_low]["logical_error_rate"]
        p_high = self._results[d_high]["logical_error_rate"]

        if p_high <= 0:
            logger.warning(
                f"p_L(d={d_high}) = 0 — cannot compute Λ. "
                f"Need more shots for statistical significance."
            )
            return float("inf")

        if p_low <= 0:
            return 0.0

        lambda_ratio = p_low / p_high
        logger.info(
            f"Λ(d={d_low}→d={d_high}) = {lambda_ratio:.4f} "
            f"({'BELOW' if lambda_ratio > 1 else 'ABOVE'} threshold)"
        )
        return lambda_ratio

    def fit_threshold_model(self) -> ThresholdFit:
        """
        Fit the phenomenological threshold model:

            p_L = A · (p / p_th)^((d+1)/2)

        to the experimental data.

        Requires results at ≥ 2 different distances.

        Returns:
            ThresholdFit with estimated p_th and A.
        """
        if len(self._results) < 2:
            raise ValueError(
                f"Need results at ≥ 2 distances for threshold fit. "
                f"Have {len(self._results)}."
            )

        distances = sorted(self._results.keys())
        error_rates = [self._results[d]["logical_error_rate"] for d in distances]
        p_phys = self._results[distances[0]]["physical_error_rate"]

        # Filter out zero error rates (can't fit log)
        valid = [(d, e) for d, e in zip(distances, error_rates) if e > 0]
        if len(valid) < 2:
            logger.warning("Too few non-zero error rates for threshold fit")
            return ThresholdFit(
                p_threshold=0.0,
                A=0.0,
                fit_residual=float("inf"),
                distances=distances,
                logical_error_rates=error_rates,
                physical_error_rate=p_phys,
                is_below_threshold=False,
            )

        d_valid = [v[0] for v in valid]
        e_valid = [v[1] for v in valid]

        # Fit: log(p_L) = log(A) + ((d+1)/2) · log(p/p_th)
        # Parameters: A, p_th
        def model(d, A, p_th):
            if p_th <= 0:
                return np.full_like(d, 1.0)
            ratio = p_phys / p_th
            if ratio <= 0:
                return np.full_like(d, 1.0)
            return A * np.power(ratio, (d + 1) / 2)

        try:
            popt, _ = optimize.curve_fit(
