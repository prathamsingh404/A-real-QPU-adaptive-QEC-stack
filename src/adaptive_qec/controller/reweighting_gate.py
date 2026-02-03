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
