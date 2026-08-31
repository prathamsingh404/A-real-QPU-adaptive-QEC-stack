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
    """Calibrates DEM edge weights using hardware measurements.

    Parameters
    ----------
    circuit : stim.Circuit, optional
        The QEC circuit (with noise) used to generate the base DEM.
        Defaults to a distance-3 rotated surface code circuit.
    scaling_mode : str
        How to scale edge probabilities:
        - "proportional": scale by ratio of measured/nominal error rates.
        - "absolute": replace with measured values directly.
    min_probability : float
        Lower clamp bound for probabilities.
    max_probability : float
        Upper clamp bound for probabilities.
    """

    def __init__(
        self,
        circuit: Optional[stim.Circuit] = None,
        scaling_mode: str = "proportional",
        min_probability: float = 1e-6,
        max_probability: float = 0.4999,
    ) -> None:
        if circuit is None:
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                distance=3,
                rounds=3,
                after_clifford_depolarization=0.003,
            )
        self._circuit = circuit
        self._scaling_mode = scaling_mode
        self._min_probability = min_probability
        self._max_probability = max_probability

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
        for instruction in circuit.flattened():
            if isinstance(instruction, stim.CircuitInstruction):
                if instruction.name == noise_type:
                    args = instruction.gate_args_copy()
                    if args:
                        rates.append(args[0])
        return float(np.mean(rates)) if rates else 0.003

    def calibrate(
        self,
        measured_p_2q: float,
        measured_p_ro: float,
        calibration_timestamp: str = "",
    ) -> CalibratedDEM:
        """Reweight the DEM using measured error rates.

        Parameters
        ----------
        measured_p_2q : float
            Measured two-qubit gate error rate.
        measured_p_ro : float
            Measured readout error rate.
        calibration_timestamp : str
            When the calibration was taken.

        Returns
        -------
        CalibratedDEM
            DEM with reweighted edges.
        """
        if self._scaling_mode == "proportional":
            return self._calibrate_proportional(
                measured_p_2q, measured_p_ro, calibration_timestamp
            )
        else:
            raise ValueError(f"Unknown scaling mode: {self._scaling_mode}")

    def _calibrate_proportional(
        self,
        measured_p_2q: float,
        measured_p_ro: float,
        calibration_timestamp: str,
    ) -> CalibratedDEM:
        """Scale edge probabilities proportionally to measured/nominal ratio."""
        ratio_2q = measured_p_2q / max(self._nominal_p_2q, 1e-10)
        ratio_ro = measured_p_ro / max(self._nominal_p_ro, 1e-10)

        # Geometric mean of the two ratios as the global scaling factor
        # (most DEM edges involve both gate and measurement operations)
        ratio = np.sqrt(ratio_2q * ratio_ro)

        new_edges: list[DEMEdge] = []
        for edge in self._base_edges:
            new_prob = np.clip(edge.probability * ratio, self._min_probability, self._max_probability)
            new_edge = DEMEdge(
                detector_a=edge.detector_a,
                detector_b=edge.detector_b,
                probability=float(new_prob),
                observables=list(edge.observables),
            )
            new_edge.compute_weight()
            new_edges.append(new_edge)

        return CalibratedDEM(
            num_detectors=self._base_dem.num_detectors,
            num_observables=self._base_dem.num_observables,
            edges=new_edges,
            calibration_timestamp=calibration_timestamp,
            source=f"proportional (ratio={ratio:.4f})",
        )

    def calibrate_per_qubit(
        self,
        qubit_error_rates: dict[int, float],
        readout_error_rates: dict[int, float],
        calibration_timestamp: str = "",
    ) -> CalibratedDEM:
        """Fine-grained reweighting using per-qubit measured errors.

        This is more accurate than global scaling but requires knowing
        which qubits each DEM edge corresponds to.  For surface codes,
        we use detector coordinates to map edges to qubits.

        Parameters
        ----------
        qubit_error_rates : dict[int, float]
            Measured gate error rate per physical qubit.
        readout_error_rates : dict[int, float]
            Measured readout error rate per physical qubit.
        """
        # For now, fall back to global calibration using the mean rates
        mean_p_2q = float(np.mean(list(qubit_error_rates.values()))) if qubit_error_rates else self._nominal_p_2q
        mean_p_ro = float(np.mean(list(readout_error_rates.values()))) if readout_error_rates else self._nominal_p_ro

        return self.calibrate(mean_p_2q, mean_p_ro, calibration_timestamp)

    def calibrate_from_syndromes(
        self,
        syndromes: np.ndarray,
        smoothing: float = 0.3,
        calibration_timestamp: str = "",
    ) -> CalibratedDEM:
        """Reweight DEM edges directly from observed sliding-window syndrome defect data.

        For each graph edge (u, v), estimates the empirical probability of
        correlated detector firing from the syndrome batch and combines it
        with the nominal base error probability via EWMA smoothing:
            p_hat = (1 - alpha) * p_base + alpha * p_empirical
            w_e = ln((1 - p_hat) / p_hat)

        Parameters
        ----------
        syndromes : np.ndarray
            Binary detection events array of shape (shots, num_detectors).
        smoothing : float
            Smoothing weight alpha in [0.0, 1.0] applied to empirical estimates.
        calibration_timestamp : str
            Timestamp metadata.

        Returns
        -------
        CalibratedDEM
            Calibrated DEM with updated edge weights via Spitz et al. (2018) parity estimator.
        """
        syndromes = np.asarray(syndromes, dtype=np.float64)
        num_shots, num_det = syndromes.shape
        if num_shots == 0:
            return CalibratedDEM(
                num_detectors=self._base_dem.num_detectors,
                num_observables=self._base_dem.num_observables,
                edges=list(self._base_edges),
                calibration_timestamp=calibration_timestamp,
                source="empty_syndromes_fallback",
            )

        mean_det = syndromes.mean(axis=0)

        # 1. Estimate internal edge probabilities via parity covariance:
        # p_ij = 0.5 - 0.5 * sqrt(1 - 4*(<xi xj> - <xi><xj>) / (1 - 2<xi> - 2<xj> + 4<xi xj>))
        internal_p: dict[tuple[int, int], float] = {}
        prod_other: dict[int, float] = {}

        for edge in self._base_edges:
            u, v = edge.detector_a, edge.detector_b
            if u >= 0 and v >= 0 and u < num_det and v < num_det:
                xij = float((syndromes[:, u] * syndromes[:, v]).mean())
                num = 4.0 * (xij - mean_det[u] * mean_det[v])
                den = 1.0 - 2.0 * mean_det[u] - 2.0 * mean_det[v] + 4.0 * xij
                val = 1.0 - num / den if abs(den) > 1e-9 else 1.0
                p_est = float(np.clip(0.5 - 0.5 * np.sqrt(max(val, 0.0)), self._min_probability, self._max_probability))
                internal_p[(min(u, v), max(u, v))] = p_est

        for (u, v), pe in internal_p.items():
            prod_other[u] = prod_other.get(u, 1.0) * (1.0 - 2.0 * pe)
            prod_other[v] = prod_other.get(v, 1.0) * (1.0 - 2.0 * pe)

        # 2. Estimate boundary edge probabilities via parity inversion:
        # 1 - 2<xi> = (1 - 2 pb) * prod_{other}(1 - 2 pe)
        new_edges: list[DEMEdge] = []
        for edge in self._base_edges:
            u, v = edge.detector_a, edge.detector_b
            if u >= 0 and v >= 0 and u < num_det and v < num_det:
                p_raw = internal_p.get((min(u, v), max(u, v)), edge.probability)
            elif (u >= 0 and v < 0 and u < num_det) or (v >= 0 and u < 0 and v < num_det):
                node = u if u >= 0 else v
                po = prod_other.get(node, 1.0)
                if abs(po) > 1e-9:
                    p_raw = 0.5 * (1.0 - (1.0 - 2.0 * mean_det[node]) / po)
                else:
                    p_raw = self._min_probability
                p_raw = float(np.clip(p_raw, self._min_probability, self._max_probability))
            else:
                p_raw = edge.probability

            # Smooth with base prior if smoothing > 0
            p_updated = (1.0 - smoothing) * p_raw + smoothing * edge.probability if smoothing > 0 else p_raw
            p_updated = float(np.clip(p_updated, self._min_probability, self._max_probability))

            new_edge = DEMEdge(
                detector_a=u,
                detector_b=v,
                probability=p_updated,
                observables=list(edge.observables),
            )
            new_edge.compute_weight()
            new_edges.append(new_edge)

        return CalibratedDEM(
            num_detectors=self._base_dem.num_detectors,
            num_observables=self._base_dem.num_observables,
            edges=new_edges,
            calibration_timestamp=calibration_timestamp,
            source=f"spitz_covariance(shots={num_shots})",
        )


def calibrate_spitz_matching(
    base_matching: pymatching.Matching,
    syndromes: np.ndarray,
    eps: float = 1e-4,
    max_p: float = 0.49,
) -> pymatching.Matching:
    """Standard analytical matching weight estimator (Spitz et al. 2018).

    Estimates edge error probabilities from syndrome covariance and returns a
    reweighted PyMatching Matching instance.
    """
    X = syndromes.astype(float)
    mean = X.mean(axis=0)
    edges = list(base_matching.edges())
    p: dict[tuple[int, int], float] = {}
    prod_other: dict[int, float] = {}

    for u, v, a in edges:
        if u is None or v is None or u < 0 or v < 0:
            continue
        xij = float((X[:, u] * X[:, v]).mean())
        num = 4.0 * (xij - mean[u] * mean[v])
        den = 1.0 - 2.0 * mean[u] - 2.0 * mean[v] + 4.0 * xij
        val = 1.0 - num / den if abs(den) > 1e-9 else 1.0
        p[(u, v)] = float(np.clip(0.5 - 0.5 * np.sqrt(max(val, 0.0)), eps, max_p))

    for (u, v), pe in p.items():
        prod_other[u] = prod_other.get(u, 1.0) * (1.0 - 2.0 * pe)
        prod_other[v] = prod_other.get(v, 1.0) * (1.0 - 2.0 * pe)

    new_matching = pymatching.Matching()
    for u, v, a in edges:
        fid = a.get("fault_ids", set())
        if u is None or v is None or u < 0 or v < 0:
            node = u if (v is None or v < 0) else v
            po = prod_other.get(node, 1.0)
            pb = 0.5 * (1.0 - (1.0 - 2.0 * mean[node]) / po) if abs(po) > 1e-9 else eps
            pb = float(np.clip(pb, eps, max_p))
            w = float(np.log((1.0 - pb) / pb))
            new_matching.add_boundary_edge(
                node,
                fault_ids=fid,
                weight=w,
                error_probability=pb,
                merge_strategy="replace",
            )
        else:
            pe = p[(u, v)]
            w = float(np.log((1.0 - pe) / pe))
            new_matching.add_edge(
                u,
                v,
                fault_ids=fid,
                weight=w,
                error_probability=pe,
                merge_strategy="replace",
            )

    return new_matching
