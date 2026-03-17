"""
Detector Error Model (DEM) calibration and reweighting.

This module builds Stim DEM edge weights from *actual* hardware
calibration data rather than using static Stim-generated models.
When hardware noise drifts, the DEM must be re-calibrated to maintain
decoder optimality.

The key insight: MWPM is optimal for the *true* noise channel.
When using outdated DEM weights, MWPM degrades. By reweighting
the DEM from fresh calibration, we recover near-optimal decoding
without retraining the decoder.

Approach:
    1. Start with the Stim-generated DEM for the circuit topology.
    2. Extract per-edge error probabilities.
    3. Update edge weights using measured gate/readout errors.
    4. Rebuild the PyMatching Matching object with new weights.

This is the "reweighting" step that makes MWPM adaptive without
changing the decoder algorithm itself.

References:
    - Spitz et al., "Adaptive weight estimator for quantum error
      correction in a time-dependent environment" (arXiv 2023)
    - Roadmap: PROJECT_EVOLUTION_ROADMAP.md §Phase-4
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pymatching
import stim

from adaptive_qec.qpu.base import CalibrationSnapshot

logger = logging.getLogger(__name__)


@dataclass
class DEMEdge:
    """A single edge in the detector error model."""
    detector_a: int          # -1 for boundary
    detector_b: int          # -1 for boundary
    probability: float       # error probability
    observables: list[int]   # which logical observables this edge flips
    weight: float = 0.0      # -log(p/(1-p)), computed from probability

    def compute_weight(self) -> float:
        """Convert probability to matching weight: w = log((1-p)/p)."""
        p = np.clip(self.probability, 1e-15, 1.0 - 1e-15)
        self.weight = float(np.log((1.0 - p) / p))
        return self.weight


@dataclass
class CalibratedDEM:
    """A detector error model with calibrated edge weights.

    Contains the original Stim DEM plus per-edge probability
    updates from hardware calibration.
    """
    num_detectors: int
    num_observables: int
    edges: list[DEMEdge]
    calibration_timestamp: str = ""
    source: str = "stim_default"

    def to_matching(self) -> pymatching.Matching:
        """Build a PyMatching Matching object from calibrated edges."""
        matching = pymatching.Matching()

        for edge in self.edges:
            if edge.detector_a < 0 and edge.detector_b < 0:
                continue

            fault_ids = set(edge.observables) if edge.observables else set()

            if edge.detector_a < 0:
                matching.add_boundary_edge(
                    edge.detector_b,
                    weight=edge.weight,
                    fault_ids=fault_ids,
                    error_probability=edge.probability,
                    merge_strategy="smallest-weight",
                )
            elif edge.detector_b < 0:
                matching.add_boundary_edge(
                    edge.detector_a,
                    weight=edge.weight,
                    fault_ids=fault_ids,
                    error_probability=edge.probability,
                    merge_strategy="smallest-weight",
                )
            else:
                matching.add_edge(
                    edge.detector_a,
                    edge.detector_b,
                    weight=edge.weight,
                    fault_ids=fault_ids,
                    error_probability=edge.probability,
                    merge_strategy="smallest-weight",
                )

        return matching


class DEMCalibrator:
