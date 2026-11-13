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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    r_squared: float = 0.0
    p_threshold_ci: tuple[float, float] = (0.0, 0.0)
    fit_method: str = "non_linear_least_squares"

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "p_threshold": round(self.p_threshold, 6),
            "A": round(self.A, 6),
            "fit_residual": round(self.fit_residual, 8),
            "r_squared": round(self.r_squared, 6),
            "p_threshold_ci": [round(x, 6) for x in self.p_threshold_ci],
            "fit_method": self.fit_method,
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

        d_valid = np.array([v[0] for v in valid], dtype=float)
        e_valid = np.array([v[1] for v in valid], dtype=float)

        # Analytic log-linear initialization:
        # ln(p_L) = ln(A) + ((d+1)/2) * ln(p_phys / p_th)
        # y = beta_0 + beta_1 * x, where x = (d+1)/2, y = ln(e)
        x_vals = (d_valid + 1.0) / 2.0
        y_vals = np.log(np.maximum(e_valid, 1e-15))

        x_mean = float(np.mean(x_vals))
        y_mean = float(np.mean(y_vals))
        s_xx = float(np.sum((x_vals - x_mean) ** 2))

        if s_xx > 1e-12:
            s_xy = float(np.sum((x_vals - x_mean) * (y_vals - y_mean)))
            beta_1 = s_xy / s_xx
            beta_0 = y_mean - beta_1 * x_mean
            A_init = float(np.clip(np.exp(beta_0), 1e-6, 10.0))
            # beta_1 = ln(p_phys / p_th) -> p_th = p_phys * exp(-beta_1)
            p_th_init = float(np.clip(p_phys * np.exp(-beta_1), 1e-5, 0.5))
        else:
            A_init = 0.1
            p_th_init = 0.01

        # Model: p_L = A * (p_phys / p_th)^((d+1)/2)
        def model(d, A, p_th):
            if p_th <= 0:
                return np.full_like(d, 1.0)
            ratio = p_phys / p_th
            if ratio <= 0:
                return np.full_like(d, 1.0)
            return A * np.power(ratio, (d + 1) / 2)

        fit_method = "non_linear_least_squares"
        pcov_diag = None

        try:
            popt, pcov = optimize.curve_fit(
                model,
                d_valid,
                e_valid,
                p0=[A_init, p_th_init],
                bounds=([1e-10, 1e-6], [10.0, 0.5]),
                maxfev=5000,
            )
            A_fit, p_th_fit = float(popt[0]), float(popt[1])
            if pcov is not None and not np.isinf(pcov).any() and pcov[1, 1] > 0:
                pcov_diag = float(pcov[1, 1])
        except (RuntimeError, ValueError) as e:
            logger.warning(f"Threshold non-linear fit failed ({e}); using analytic log-linear fit.")
            A_fit = A_init
            p_th_fit = p_th_init
            fit_method = "analytic_log_linear"

        # Compute residuals and R-squared
        predicted = model(d_valid, A_fit, p_th_fit)
        rss = float(np.sum((e_valid - predicted) ** 2))
        tss = float(np.sum((e_valid - np.mean(e_valid)) ** 2))
        r_squared = float(np.clip(1.0 - rss / max(tss, 1e-15), 0.0, 1.0))

        # 95% Confidence Interval for p_threshold
        if pcov_diag is not None:
            se_pth = float(np.sqrt(pcov_diag))
        else:
            df = max(len(d_valid) - 2, 1)
            se_pth = float(np.sqrt(rss / df))
        ci_pth = (max(0.0, float(p_th_fit - 1.96 * se_pth)), float(p_th_fit + 1.96 * se_pth))

        # Compute all lambda ratios
        lambda_ratios = {}
        for i in range(len(distances) - 1):
            d_lo, d_hi = distances[i], distances[i + 1]
            try:
                lam = self.compute_lambda(d_lo, d_hi)
                lambda_ratios[f"d{d_lo}_to_d{d_hi}"] = lam
            except ValueError:
                pass

        is_below = all(v > 1.0 for v in lambda_ratios.values()) if lambda_ratios else False

        fit = ThresholdFit(
            p_threshold=float(p_th_fit),
            A=float(A_fit),
            fit_residual=rss,
            distances=distances,
            logical_error_rates=error_rates,
            physical_error_rate=p_phys,
            lambda_ratios=lambda_ratios,
            is_below_threshold=is_below,
            r_squared=r_squared,
            p_threshold_ci=ci_pth,
            fit_method=fit_method,
        )

        logger.info(
            f"Threshold fit: p_th={p_th_fit:.6f}, A={A_fit:.4f}, "
            f"R2={r_squared:.4f}, below_threshold={is_below}, Λ={lambda_ratios}"
        )
        return fit

    @staticmethod
    def find_threshold_crossing(
        distances: Sequence[int],
        physical_error_rates: Sequence[float],
        logical_error_matrix: np.ndarray | Sequence[Sequence[float]],
    ) -> Optional[float]:
        """
        Find the empirical threshold crossing point where distance curves intersect.

        Below threshold, larger distance achieves lower logical error rate (d2 < d1).
        Above threshold, larger distance achieves higher logical error rate (d2 > d1).
        At threshold, curves intersect: delta(p_th) = 0.

        Args:
            distances: Sequence of code distances [d1, d2, ...] in increasing order.
            physical_error_rates: Sequence of physical error rates in increasing order.
            logical_error_matrix: Shape (len(distances), len(physical_error_rates))
                matrix of logical error rates.

        Returns:
            float: Estimated physical error rate at crossing point p_th, or None.
        """
        mat = np.asarray(logical_error_matrix, dtype=np.float64)
        p_arr = np.asarray(physical_error_rates, dtype=np.float64)
        num_d = len(distances)

        if num_d < 2 or len(p_arr) < 2:
            return None

        crossings: list[float] = []

        # Check consecutive distance pairs (d_i, d_{i+1})
        for i in range(num_d - 1):
            ler_lo = mat[i, :]
            ler_hi = mat[i + 1, :]
            delta = ler_hi - ler_lo  # negative below threshold, positive above

            # Look for sign changes where delta crosses zero
            for j in range(len(p_arr) - 1):
                d_curr = delta[j]
                d_next = delta[j + 1]

                if d_curr == 0.0:
                    crossings.append(float(p_arr[j]))
                elif (d_curr < 0.0 < d_next) or (d_curr > 0.0 > d_next):
                    # Linear interpolation of the zero crossing
                    slope = (d_next - d_curr) / (p_arr[j + 1] - p_arr[j])
                    if abs(slope) > 1e-12:
                        p_cross = float(p_arr[j] - d_curr / slope)
                        crossings.append(p_cross)

        if not crossings:
            return None

        return float(np.mean(crossings))

    def get_results(self) -> dict[int, dict[str, Any]]:
        """Get all collected results."""
        return dict(self._results)

    def scaling_table(self) -> list[dict[str, Any]]:
        """
        Generate a scaling table for display/export.

        Returns list of dicts with distance, error rate, CI, and Λ.
        """
        distances = sorted(self._results.keys())
        table = []

        for i, d in enumerate(distances):
            r = self._results[d]
            row = {
                "distance": d,
                "logical_error_rate": r["logical_error_rate"],
                "ci_lower": r["ci_lower"],
                "ci_upper": r["ci_upper"],
                "num_errors": r["num_logical_errors"],
                "total_shots": r["total_shots"],
            }

            # Lambda to previous distance
            if i > 0:
                d_prev = distances[i - 1]
                p_prev = self._results[d_prev]["logical_error_rate"]
                p_curr = r["logical_error_rate"]
                row["lambda_vs_prev"] = p_prev / p_curr if p_curr > 0 else float("inf")
            else:
                row["lambda_vs_prev"] = None

            table.append(row)

        return table
