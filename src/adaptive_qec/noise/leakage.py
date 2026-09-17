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
