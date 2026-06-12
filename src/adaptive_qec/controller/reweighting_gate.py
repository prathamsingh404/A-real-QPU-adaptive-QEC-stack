"""
Label-Free Confidence-Gated Decoder Reweighting Policy (CG-CCADS).

Operates strictly within deployable, online observation boundaries:
- ZERO access to ground-truth logical observable labels.
- Ingests sliding-window detection events (syndromes).
- Evaluates statistical evidence of localized physical drift using Spitz et al. (2018)
  covariance-parity estimators against finite-sample shot noise and adaptation taxes.
- Applies Bonferroni family-wise error rate (FWER) correction across matching edges.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple

import numpy as np
import pymatching
from scipy import stats

from adaptive_qec.decoders.dem_calibration import calibrate_spitz_matching

logger = logging.getLogger(__name__)


class GateDecision(str, Enum):
    """Outcomes of the confidence-gated reweighting pipeline."""
    STAY = "stay"          # Abstain: keep current matching graph (0 cost)
    REWEIGHT = "reweight"  # Update matching graph to Spitz-estimated weights
    RESET = "reset"        # Revert to factory baseline prior


@dataclass
class GateStatistics:
    """Label-free diagnostic statistics computed from syndrome window."""
    window_size: int
    mean_defect_rate: float
    drift_score_z: float
    max_edge_shift: float
    max_edge_lcb: float
    decision: GateDecision
    reason: str


class ConfidenceGatedReweightingController:
    """
    Confidence-Gated Cost-Constrained Reweighting Controller for MWPM.
    
    Enforces a rigorous 3-gate statistical barrier before reweighting:
      Gate 1 (Stationarity): Rejects adaptation if overall detector firing rates
             are statistically consistent with the baseline stationary reference.
      Gate 2 (Deadzone / Noise Floor): Computes the Lower Confidence Bound (LCB)
             of edge probability shifts under finite sample size W with Bonferroni
             correction across all candidate edges. If LCB <= deadzone,
             the apparent shift is consistent with finite-sample binomial variance -> STAY.
      Gate 3 (Adaptation Tax): Reweights only if LCB exceeds the physical and
             computational switching tax (tau_adapt).
    """

    def __init__(
        self,
        base_matching: pymatching.Matching,
        alpha_significance: float = 0.05,
        drift_z_threshold: float = 2.5,
        adaptation_tax: float = 0.003,
        dead_zone_epsilon: float = 0.002,
        reference_defect_rate: Optional[float] = None,
    ) -> None:
        self.base_matching = base_matching
        self.alpha_significance = alpha_significance
        self.drift_z_threshold = drift_z_threshold
        self.adaptation_tax = adaptation_tax
        self.dead_zone_epsilon = dead_zone_epsilon

        # Extract base edges
        self._reference_edges: dict[tuple[int, int], float] = self._extract_matching_probabilities(base_matching)
        num_edges = max(len(self._reference_edges), 1)

        # Bonferroni corrected normal quantile for family-wise error rate alpha across edges
        self.z_crit = float(stats.norm.ppf(1.0 - alpha_significance / (2.0 * num_edges)))

        # Active matching graph
        self.active_matching: pymatching.Matching = base_matching
        self.is_reweighted: bool = False

        # Baseline reference stats
        self.reference_defect_rate = reference_defect_rate

        # Telemetry counters
        self.total_stays: int = 0
        self.total_reweights: int = 0
        self.total_resets: int = 0
        self.last_stats: Optional[GateStatistics] = None

    @staticmethod
    def _extract_matching_probabilities(matching: pymatching.Matching) -> dict[tuple[int, int], float]:
        """Extract edge error probabilities from a PyMatching graph."""
        probs: dict[tuple[int, int], float] = {}
        for u, v, data in matching.edges():
            p = data.get("error_probability", 0.0)
            if p is None or p <= 0.0:
                w = data.get("weight", 1.0)
                # w = ln((1-p)/p) -> p = 1 / (1 + exp(w))
                p = float(1.0 / (1.0 + np.exp(np.clip(w, -20.0, 20.0))))
            u_node = -1 if (u is None or u < 0) else int(u)
            v_node = -1 if (v is None or v < 0) else int(v)
            key = (min(u_node, v_node), max(u_node, v_node))
            probs[key] = float(p)
        return probs

    def evaluate_window(
        self,
        syndromes: np.ndarray,
    ) -> Tuple[GateDecision, pymatching.Matching, GateStatistics]:
        """
        Evaluate a syndrome window and decide whether to STAY, REWEIGHT, or RESET.

        Parameters
        ----------
        syndromes : np.ndarray
            Binary array of shape (W, num_detectors).

        Returns
        -------
        Tuple[GateDecision, pymatching.Matching, GateStatistics]
        """
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        w_size, num_det = syndromes.shape
        if w_size == 0 or num_det == 0:
            stats_obj = GateStatistics(w_size, 0.0, 0.0, 0.0, 0.0, GateDecision.STAY, "empty_window")
            return GateDecision.STAY, self.active_matching, stats_obj

        # Shot-level defect rates to capture inter-detector covariance accurately
        shot_defect_rates = syndromes.mean(axis=1)
        mean_defect = float(shot_defect_rates.mean())

        # Initialize reference defect rate if not provided
        if self.reference_defect_rate is None:
            self.reference_defect_rate = mean_defect

        # Standard error of the mean shot defect rate
        se_defect = float(np.std(shot_defect_rates, ddof=1) / np.sqrt(w_size)) if w_size > 1 else 0.01
        z_drift = abs(mean_defect - self.reference_defect_rate) / max(se_defect, 1e-6)
