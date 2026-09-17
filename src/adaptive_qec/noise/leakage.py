"""
Leakage detection from QEC syndrome data.

Leakage — a qubit transitioning from the computational subspace {|0⟩, |1⟩}
to higher energy states {|2⟩, |3⟩, ...} — is a persistent, correlated error
that standard QEC can't handle. A leaked qubit produces incorrect syndrome
information for *every subsequent round* until it's detected and reset.

Detection approach:
    A leaked qubit produces a *persistent* defect pattern: the same detector
    fires every round (or nearly every round). This is distinct from a
    transient error, which produces defects in isolated rounds.

    We detect leakage by computing the temporal autocorrelation of each
    detector's firing history. High autocorrelation at lag 1 indicates
    a persistent defect — the signature of leakage.

Sources:
    - Google AlphaQubit — leakage handling via neural network decoder
    - IBM Heron r2 — native leakage reduction circuits
    - Battistel et al., "Hardware-efficient leakage-reduction scheme for
      quantum error correction with superconducting transmon qubits" (2021)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LeakedQubit:
    """A detector suspected of being affected by leakage."""
    detector_index: int
    onset_round: int              # Estimated round when leakage started
    persistence_length: int       # Number of consecutive rounds with defects
    autocorrelation: float        # Temporal autocorrelation at lag 1
    firing_rate: float            # Fraction of rounds this detector fired
    confidence: float             # Leakage confidence [0, 1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_index": self.detector_index,
            "onset_round": self.onset_round,
            "persistence_length": self.persistence_length,
            "autocorrelation": round(self.autocorrelation, 4),
            "firing_rate": round(self.firing_rate, 4),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class LeakageAnalysis:
    """Complete leakage analysis results."""
    total_rounds: int
    total_detectors: int
    leaked_qubits: list[LeakedQubit] = field(default_factory=list)
    estimated_leakage_rate: float = 0.0  # leakage events per round per qubit

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rounds": self.total_rounds,
            "total_detectors": self.total_detectors,
            "num_leaked": len(self.leaked_qubits),
            "estimated_leakage_rate": round(self.estimated_leakage_rate, 8),
            "leaked_qubits": [q.to_dict() for q in self.leaked_qubits],
        }


class LeakageDetector:
    """
    Detects leakage from QEC syndrome temporal patterns.

    A leaked qubit produces persistent, round-after-round defects on
    the same detector. We identify these by:
        1. Computing per-detector firing rate
        2. Computing temporal autocorrelation at lag 1
        3. Finding the longest consecutive firing streak
        4. Combining these signals into a leakage confidence score

    Usage:
        detector = LeakageDetector(min_persistence=3, autocorr_threshold=0.3)
        analysis = detector.analyze(syndrome_tensor)
    """

    def __init__(
        self,
        min_persistence: int = 3,
        autocorr_threshold: float = 0.3,
        firing_rate_threshold: float = 0.4,
    ) -> None:
        """
        Args:
            min_persistence: Min consecutive rounds for leakage suspicion.
            autocorr_threshold: Min lag-1 autocorrelation for leakage.
            firing_rate_threshold: Min firing rate for leakage suspicion.
        """
        self.min_persistence = min_persistence
        self.autocorr_threshold = autocorr_threshold
        self.firing_rate_threshold = firing_rate_threshold

    def analyze(self, syndrome_tensor: np.ndarray) -> LeakageAnalysis:
        """
        Analyze a syndrome tensor for leakage signatures.

        Args:
            syndrome_tensor: shape (rounds, detectors_per_round), dtype uint8.

        Returns:
            LeakageAnalysis with detected leaked qubits.
        """
        R, N_d = syndrome_tensor.shape
        leaked: list[LeakedQubit] = []

        for det_idx in range(N_d):
            trace = syndrome_tensor[:, det_idx].astype(np.float64)
            firing_rate = float(trace.mean())

            # Skip detectors that rarely fire
            if firing_rate < self.firing_rate_threshold:
                continue

            # Temporal autocorrelation at lag 1
            autocorr = self._autocorrelation_lag1(trace)

            # Longest consecutive firing streak
            persistence, onset = self._longest_streak(trace)

            # Skip if persistence is too short
            if persistence < self.min_persistence:
                continue

            # Skip if autocorrelation is too low
            if autocorr < self.autocorr_threshold:
                continue

            # Compute leakage confidence
            confidence = self._compute_confidence(
                autocorr=autocorr,
                persistence=persistence,
                firing_rate=firing_rate,
                total_rounds=R,
            )

            leaked.append(LeakedQubit(
                detector_index=det_idx,
                onset_round=onset,
                persistence_length=persistence,
                autocorrelation=autocorr,
                firing_rate=firing_rate,
                confidence=confidence,
            ))

            logger.info(
                f"Leakage suspected: detector {det_idx}, "
                f"onset round {onset}, persistence={persistence}, "
                f"autocorr={autocorr:.3f}, confidence={confidence:.3f}"
            )

        # Estimate leakage rate
        if R > 0 and N_d > 0:
            leakage_rate = len(leaked) / (R * N_d)
        else:
            leakage_rate = 0.0

        return LeakageAnalysis(
            total_rounds=R,
            total_detectors=N_d,
            leaked_qubits=leaked,
            estimated_leakage_rate=leakage_rate,
        )

    @staticmethod
    def _autocorrelation_lag1(trace: np.ndarray) -> float:
        """Compute lag-1 autocorrelation of a binary time series."""
        n = len(trace)
        if n < 2:
            return 0.0

        mean = trace.mean()
        var = trace.var()
        if var < 1e-10:
            return 0.0 if mean < 0.5 else 1.0

        shifted = trace[1:] - mean
        original = trace[:-1] - mean
        autocorr = float(np.sum(shifted * original) / ((n - 1) * var))
        return np.clip(autocorr, -1.0, 1.0)

    @staticmethod
    def _longest_streak(trace: np.ndarray) -> tuple[int, int]:
        """
        Find the longest consecutive run of 1s.

        Returns:
            (streak_length, onset_index)
        """
        max_streak = 0
        max_onset = 0
        current_streak = 0
        current_onset = 0

        for i, val in enumerate(trace):
            if val > 0:
                if current_streak == 0:
                    current_onset = i
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
                    max_onset = current_onset
            else:
                current_streak = 0

        return max_streak, max_onset

    @staticmethod
    def _compute_confidence(
        autocorr: float,
        persistence: int,
        firing_rate: float,
        total_rounds: int,
    ) -> float:
        """
        Compute leakage confidence from multiple signals.

        High confidence requires:
            - High autocorrelation (persistent same-round defects)
            - Long consecutive streak
            - High firing rate
        """
        # Normalize each signal to [0, 1]
        autocorr_score = np.clip(autocorr, 0, 1)
        persistence_score = min(persistence / max(total_rounds * 0.5, 1), 1.0)
        rate_score = min(firing_rate / 0.8, 1.0)

        # Weighted combination
        confidence = (
            0.4 * autocorr_score +
            0.35 * persistence_score +
            0.25 * rate_score
        )
        return float(np.clip(confidence, 0, 1))


class LeakageRateEstimator:
    """
    Estimate per-qubit leakage and seepage-back rates from repeated
    experiments.

    Leakage rate (γ_L): probability per round of transitioning |0⟩/|1⟩ → |2⟩
    Seepage rate (γ_S): probability per round of transitioning |2⟩ → |0⟩/|1⟩

    These rates govern the steady-state leaked population:
        p_leak_steady = γ_L / (γ_L + γ_S)
    """

    def __init__(self) -> None:
        self._leakage_events: list[float] = []
        self._seepage_events: list[float] = []
        self._total_rounds: int = 0
        self._total_qubits: int = 0

    def add_analysis(self, analysis: LeakageAnalysis) -> None:
        """Add results from a single experiment."""
        self._total_rounds += analysis.total_rounds
        self._total_qubits += analysis.total_detectors

        for lq in analysis.leaked_qubits:
            self._leakage_events.append(lq.persistence_length)

    def estimate_rates(self) -> dict[str, float]:
        """
        Estimate leakage and seepage rates.

        Returns:
            Dict with gamma_leakage, gamma_seepage, p_leak_steady.
        """
        if self._total_rounds == 0 or self._total_qubits == 0:
            return {
                "gamma_leakage": 0.0,
                "gamma_seepage": 0.0,
