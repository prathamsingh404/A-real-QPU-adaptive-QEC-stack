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
        matching.set_boundary_nodes({-1})

        for edge in self.edges:
            if edge.detector_a < 0 and edge.detector_b < 0:
                continue

            fault_ids = edge.observables if edge.observables else None

            if edge.detector_a < 0:
                matching.add_boundary_edge(
                    edge.detector_b,
                    weight=edge.weight,
                    fault_ids=fault_ids,
                    error_probability=edge.probability,
                )
            elif edge.detector_b < 0:
                matching.add_boundary_edge(
                    edge.detector_a,
                    weight=edge.weight,
                    fault_ids=fault_ids,
                    error_probability=edge.probability,
                )
            else:
                matching.add_edge(
                    edge.detector_a,
                    edge.detector_b,
                    weight=edge.weight,
                    fault_ids=fault_ids,
                    error_probability=edge.probability,
                )

        return matching


class DEMCalibrator:
    """Calibrates DEM edge weights using hardware measurements.

    Parameters
    ----------
    circuit : stim.Circuit
        The QEC circuit (with noise) used to generate the base DEM.
    scaling_mode : str
        How to scale edge probabilities:
        - "proportional": scale by ratio of measured/nominal error rates.
        - "absolute": replace with measured values directly.
    """

    def __init__(
        self,
        circuit: stim.Circuit,
        scaling_mode: str = "proportional",
    ) -> None:
        self._circuit = circuit
        self._scaling_mode = scaling_mode

        # Extract base DEM
        self._base_dem = circuit.detector_error_model(decompose_errors=True)
        self._base_edges = self._parse_dem(self._base_dem)

        # Store nominal error rates from the circuit
        self._nominal_p_2q = self._extract_circuit_noise(circuit, "DEPOLARIZE2")
        self._nominal_p_ro = self._extract_circuit_noise(circuit, "X_ERROR")

    def _parse_dem(self, dem: stim.DetectorErrorModel) -> list[DEMEdge]:
        """Parse a Stim DEM into a list of DEMEdge objects."""
        edges: list[DEMEdge] = []

        for instruction in dem.flattened():
            if not isinstance(instruction, stim.DemInstruction):
                continue
            if instruction.type != "error":
                continue

            probability = instruction.args_copy()[0]
            detectors: list[int] = []
            observables: list[int] = []

            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    detectors.append(target.val)
                elif target.is_logical_observable_id():
                    observables.append(target.val)

            if len(detectors) == 2:
                edge = DEMEdge(
                    detector_a=detectors[0],
                    detector_b=detectors[1],
                    probability=probability,
                    observables=observables,
                )
            elif len(detectors) == 1:
                edge = DEMEdge(
                    detector_a=detectors[0],
                    detector_b=-1,
                    probability=probability,
                    observables=observables,
                )
            elif len(detectors) == 0 and observables:
                edge = DEMEdge(
                    detector_a=-1,
                    detector_b=-1,
                    probability=probability,
                    observables=observables,
                )
            else:
                continue

            edge.compute_weight()
            edges.append(edge)

        return edges

    def _extract_circuit_noise(self, circuit: stim.Circuit, noise_type: str) -> float:
        """Extract the dominant noise rate of a given type from the circuit."""
        rates: list[float] = []
