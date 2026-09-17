"""
Correlated error burst detection for QEC syndromes.

Detects spatiotemporal clusters of errors that violate the independent-error
assumption underlying surface codes. Caused by:
    - Cosmic ray impacts (wide spatial, sharp temporal)
    - Quasiparticle poisoning (localized, lingering)
    - Crosstalk events (patterned, gate-correlated)

These bursts are rare (~1/hour) but catastrophic: they can cause correlated
logical failures that defeat error correction.

Detection approach:
    For each time window of width w, compute whether the observed defect
    rate exceeds the null hypothesis (independent Bernoulli at rate p_base)
    using a chi-squared-like test at significance level α.

Sources:
    - Google Quantum AI, "Quantum error correction below the surface code
      threshold" (2025) — cosmic ray discussion
    - McEwen et al., "Resolving catastrophic error bursts from cosmic rays
      in large arrays of superconducting qubits" (2022)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class BurstType(str, Enum):
    """Classification of detected error bursts."""
    COSMIC_RAY = "cosmic_ray"       # Wide spatial, sharp temporal
    QP_POISONING = "qp_poisoning"   # Localized, lingering
    CROSSTALK = "crosstalk"         # Patterned, gate-correlated
    UNKNOWN = "unknown"


@dataclass
class BurstEvent:
    """A detected error burst."""
    round_start: int           # First round of the burst
    round_end: int             # Last round of the burst
    affected_detectors: list[int]  # Detector indices involved
    severity: float            # Chi-squared test statistic
    p_value: float             # p-value under null hypothesis
    burst_type: BurstType      # Heuristic classification
    spatial_radius: float      # Estimated spatial extent
    confidence: float          # Classification confidence [0, 1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_start": self.round_start,
            "round_end": self.round_end,
            "affected_detectors": self.affected_detectors,
            "severity": round(self.severity, 4),
            "p_value": round(self.p_value, 8),
            "burst_type": self.burst_type.value,
            "spatial_radius": round(self.spatial_radius, 2),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class BurstAnalysis:
    """Complete burst analysis results."""
    total_rounds: int
    total_detectors: int
    baseline_defect_rate: float
    bursts_detected: list[BurstEvent] = field(default_factory=list)
    burst_rate_per_round: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rounds": self.total_rounds,
            "total_detectors": self.total_detectors,
            "baseline_defect_rate": round(self.baseline_defect_rate, 6),
            "num_bursts": len(self.bursts_detected),
            "burst_rate_per_round": round(self.burst_rate_per_round, 8),
            "bursts": [b.to_dict() for b in self.bursts_detected],
        }


class BurstDetector:
    """
    Spatiotemporal burst detector for QEC syndrome data.

    Scans the syndrome tensor S ∈ {0,1}^(R × N_d) for anomalous
    spatiotemporal clusters using a sliding-window chi-squared test.

    Usage:
        detector = BurstDetector(window_size=5, significance=0.001)
        analysis = detector.analyze(syndrome_tensor, num_detectors_per_round=8)
    """

    def __init__(
        self,
        window_size: int = 5,
        significance: float = 0.001,
        min_defects_for_burst: int = 3,
    ) -> None:
        """
        Args:
            window_size: Number of rounds in the sliding window.
            significance: p-value threshold for burst detection (α).
            min_defects_for_burst: Minimum defects in a window to trigger.
        """
        self.window_size = window_size
        self.significance = significance
        self.min_defects = min_defects_for_burst

    def analyze(
        self,
        syndrome_tensor: np.ndarray,
        num_detectors_per_round: int,
    ) -> BurstAnalysis:
        """
        Analyze a syndrome tensor for error bursts.

        Args:
            syndrome_tensor: shape (rounds, detectors_per_round), dtype uint8.
                Each entry is 1 if the detector fired, 0 otherwise.
            num_detectors_per_round: Number of detectors per QEC round.

        Returns:
            BurstAnalysis with detected bursts.
        """
        R, N_d = syndrome_tensor.shape
        assert N_d == num_detectors_per_round, (
            f"Expected {num_detectors_per_round} detectors/round, got {N_d}"
        )

        # Compute baseline defect rate (excluding obvious outlier rounds)
        round_rates = syndrome_tensor.mean(axis=1)
        median_rate = float(np.median(round_rates))
        baseline_rate = max(median_rate, 1e-6)

        # Expected defects per window under null hypothesis
        expected_per_window = baseline_rate * N_d * self.window_size

        # Sliding window burst detection
        bursts: list[BurstEvent] = []
        w = self.window_size

        round_idx = 0
        while round_idx <= R - w:
            window = syndrome_tensor[round_idx:round_idx + w]
            total_defects = int(window.sum())

            if total_defects < self.min_defects:
                round_idx += 1
                continue

            # Chi-squared test: is observed defect count significantly
            # above expected under independent Bernoulli model?
            # H0: defects ~ Binomial(N_d * w, baseline_rate)
            # Use Poisson approximation for large N_d * w
            if expected_per_window > 0:
                # One-sided Poisson test
                p_value = 1.0 - stats.poisson.cdf(
                    total_defects - 1, expected_per_window
                )
            else:
