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
        num_edges = len(self._reference_edges)

        # Guard: when num_edges == 0, no Bonferroni correction needed
        if num_edges == 0:
            self.z_crit = float(stats.norm.ppf(1.0 - alpha_significance / 2.0))
        else:
            # Bonferroni corrected normal quantile for family-wise error rate alpha across edges
            self.z_crit = float(stats.norm.ppf(1.0 - alpha_significance / (2.0 * num_edges)))

        # Active matching graph
        self.active_matching: pymatching.Matching = base_matching
        self.is_reweighted: bool = False

        # Baseline reference stats
        self.reference_defect_rate = reference_defect_rate
        self.reference_shot_defect_rates: Optional[np.ndarray] = None

        # Telemetry counters
        self.total_stays: int = 0
        self.total_reweights: int = 0
        self.total_resets: int = 0
        self.last_stats: Optional[GateStatistics] = None

    def reset_reference(
        self,
        new_reference_rate: Optional[float] = None,
        new_reference_samples: Optional[np.ndarray] = None,
    ) -> None:
        """Reset the reference defect rate baseline and optional reference samples."""
        self.reference_defect_rate = new_reference_rate
        self.reference_shot_defect_rates = (
            np.asarray(new_reference_samples, dtype=np.float64)
            if new_reference_samples is not None
            else None
        )

    def get_telemetry(self) -> dict[str, Any]:
        """Return a complete telemetry snapshot of controller state and decisions."""
        return {
            "total_stays": self.total_stays,
            "total_reweights": self.total_reweights,
            "total_resets": self.total_resets,
            "is_reweighted": self.is_reweighted,
            "reference_defect_rate": self.reference_defect_rate,
            "z_crit": self.z_crit,
            "drift_z_threshold": self.drift_z_threshold,
            "adaptation_tax": self.adaptation_tax,
            "dead_zone_epsilon": self.dead_zone_epsilon,
            "last_stats": {
                "window_size": self.last_stats.window_size,
                "mean_defect_rate": self.last_stats.mean_defect_rate,
                "drift_score_z": self.last_stats.drift_score_z,
                "max_edge_shift": self.last_stats.max_edge_shift,
                "max_edge_lcb": self.last_stats.max_edge_lcb,
                "decision": self.last_stats.decision.value,
                "reason": self.last_stats.reason,
            } if self.last_stats is not None else None,
        }

    def serialize_state(self) -> dict[str, Any]:
        """Serialize controller configuration, counters, and reference state to dict."""
        return {
            "alpha_significance": self.alpha_significance,
            "drift_z_threshold": self.drift_z_threshold,
            "adaptation_tax": self.adaptation_tax,
            "dead_zone_epsilon": self.dead_zone_epsilon,
            "reference_defect_rate": self.reference_defect_rate,
            "reference_shot_defect_rates": (
                self.reference_shot_defect_rates.tolist()
                if self.reference_shot_defect_rates is not None
                else None
            ),
            "is_reweighted": self.is_reweighted,
            "total_stays": self.total_stays,
            "total_reweights": self.total_reweights,
            "total_resets": self.total_resets,
            "z_crit": self.z_crit,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore controller state from serialized dictionary."""
        self.alpha_significance = state.get("alpha_significance", self.alpha_significance)
        self.drift_z_threshold = state.get("drift_z_threshold", self.drift_z_threshold)
        self.adaptation_tax = state.get("adaptation_tax", self.adaptation_tax)
        self.dead_zone_epsilon = state.get("dead_zone_epsilon", self.dead_zone_epsilon)
        self.reference_defect_rate = state.get("reference_defect_rate", self.reference_defect_rate)
        ref_shots = state.get("reference_shot_defect_rates")
        self.reference_shot_defect_rates = np.array(ref_shots, dtype=np.float64) if ref_shots is not None else None
        self.is_reweighted = bool(state.get("is_reweighted", False))
        self.total_stays = int(state.get("total_stays", 0))
        self.total_reweights = int(state.get("total_reweights", 0))
        self.total_resets = int(state.get("total_resets", 0))
        if "z_crit" in state:
            self.z_crit = float(state["z_crit"])
        if not self.is_reweighted:
            self.active_matching = self.base_matching

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
            self.reference_shot_defect_rates = shot_defect_rates.copy()
            z_drift = 0.0
        elif self.reference_shot_defect_rates is not None and len(self.reference_shot_defect_rates) > 1 and w_size > 1:
            # Proper Welch's t-test for stationarity check when window sizes differ
            t_res = stats.ttest_ind(
                shot_defect_rates,
                self.reference_shot_defect_rates,
                equal_var=False,
            )
            t_stat = float(t_res.statistic)
            z_drift = abs(t_stat) if np.isfinite(t_stat) else 0.0
        else:
            # Standard error of the mean shot defect rate under scalar reference
            se_defect = float(np.std(shot_defect_rates, ddof=1) / np.sqrt(w_size)) if w_size > 1 else 0.01
            z_drift = abs(mean_defect - self.reference_defect_rate) / max(se_defect, 1e-6)

        # Gate 1: Stationarity Check
        if z_drift <= self.drift_z_threshold:
            if self.is_reweighted:
                # Drift has subsided: reset to factory baseline
                self.is_reweighted = False
                self.active_matching = self.base_matching
                self.total_resets += 1
                stats_obj = GateStatistics(
                    w_size, mean_defect, z_drift, 0.0, 0.0,
                    GateDecision.RESET, "gate1_drift_subsided_reset_to_base"
                )
                self.last_stats = stats_obj
                return GateDecision.RESET, self.active_matching, stats_obj

            self.total_stays += 1
            stats_obj = GateStatistics(
                w_size, mean_defect, z_drift, 0.0, 0.0,
                GateDecision.STAY, "gate1_stationary_stay"
            )
            self.last_stats = stats_obj
            return GateDecision.STAY, self.active_matching, stats_obj

        # Drift detected: compute Spitz edge probability estimates from window
        candidate_matching = calibrate_spitz_matching(self.base_matching, syndromes)
        candidate_edges = self._extract_matching_probabilities(candidate_matching)

        # Compute max Bonferroni LCB on edge shifts
        max_shift = 0.0
        max_lcb = -float("inf")

        for key, p_cand in candidate_edges.items():
            p_ref = self._reference_edges.get(key, 0.004)
            shift = abs(p_cand - p_ref)
            if shift > max_shift:
                max_shift = shift

            # Conservative standard error under the null reference probability
            p_se = max(p_ref, p_cand, 1e-4)
            se_edge = np.sqrt(p_se * (1.0 - p_se) / max(w_size, 1))
            lcb_edge = shift - self.z_crit * se_edge
            if lcb_edge > max_lcb:
                max_lcb = lcb_edge

        # Gate 2: Noise Floor / Deadzone Check
        if max_lcb <= self.dead_zone_epsilon:
            self.total_stays += 1
            stats_obj = GateStatistics(
                w_size, mean_defect, z_drift, max_shift, max_lcb,
                GateDecision.STAY, f"gate2_noise_floor_stay(lcb={max_lcb:.4f}<=deadzone={self.dead_zone_epsilon:.4f})"
            )
            self.last_stats = stats_obj
            return GateDecision.STAY, self.active_matching, stats_obj

        # Gate 3: Cost-Constrained Threshold Check
        required_barrier = self.adaptation_tax + self.dead_zone_epsilon
        if max_lcb > required_barrier:
            self.active_matching = candidate_matching
            self.is_reweighted = True
            self.total_reweights += 1
            stats_obj = GateStatistics(
                w_size, mean_defect, z_drift, max_shift, max_lcb,
                GateDecision.REWEIGHT, f"gate3_confidence_switch(lcb={max_lcb:.4f}>tax={required_barrier:.4f})"
            )
            self.last_stats = stats_obj
            return GateDecision.REWEIGHT, self.active_matching, stats_obj

        # Fallback: STAY (cost barrier not cleared)
        self.total_stays += 1
        stats_obj = GateStatistics(
            w_size, mean_defect, z_drift, max_shift, max_lcb,
            GateDecision.STAY, f"gate3_tax_barrier_stay(lcb={max_lcb:.4f}<=tax={required_barrier:.4f})"
        )
        self.last_stats = stats_obj
        return GateDecision.STAY, self.active_matching, stats_obj

    def get_telemetry(self) -> dict[str, Any]:
        """Export controller telemetry counters and operational state."""
        total_evals = self.total_stays + self.total_reweights + self.total_resets
        abstention_rate = (self.total_stays / total_evals) if total_evals > 0 else 1.0
        return {
            "total_evaluations": total_evals,
            "total_stays": self.total_stays,
            "total_reweights": self.total_reweights,
            "total_resets": self.total_resets,
            "abstention_rate": float(abstention_rate),
            "is_reweighted": self.is_reweighted,
            "reference_defect_rate": self.reference_defect_rate,
            "drift_z_threshold": self.drift_z_threshold,
            "adaptation_tax": self.adaptation_tax,
            "dead_zone_epsilon": self.dead_zone_epsilon,
            "last_decision": self.last_stats.decision.value if self.last_stats else None,
            "last_reason": self.last_stats.reason if self.last_stats else None,
        }

    def reset_reference(self, new_reference: Optional[float] = None) -> None:
        """Manually re-anchor or reset the baseline reference defect rate."""
        self.reference_defect_rate = new_reference
        self.active_matching = self.base_matching
        self.is_reweighted = False

    def serialize_state(self) -> dict[str, Any]:
        """Serialize configuration and internal telemetry to dictionary."""
        return {
            "alpha_significance": self.alpha_significance,
            "drift_z_threshold": self.drift_z_threshold,
            "adaptation_tax": self.adaptation_tax,
            "dead_zone_epsilon": self.dead_zone_epsilon,
            "reference_defect_rate": self.reference_defect_rate,
            "total_stays": self.total_stays,
            "total_reweights": self.total_reweights,
            "total_resets": self.total_resets,
            "is_reweighted": self.is_reweighted,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore controller counters and reference values from serialized dictionary."""
        self.alpha_significance = float(state.get("alpha_significance", self.alpha_significance))
        self.drift_z_threshold = float(state.get("drift_z_threshold", self.drift_z_threshold))
        self.adaptation_tax = float(state.get("adaptation_tax", self.adaptation_tax))
        self.dead_zone_epsilon = float(state.get("dead_zone_epsilon", self.dead_zone_epsilon))
        self.reference_defect_rate = state.get("reference_defect_rate", self.reference_defect_rate)
        self.total_stays = int(state.get("total_stays", self.total_stays))
        self.total_reweights = int(state.get("total_reweights", self.total_reweights))
        self.total_resets = int(state.get("total_resets", self.total_resets))
        self.is_reweighted = bool(state.get("is_reweighted", self.is_reweighted))
