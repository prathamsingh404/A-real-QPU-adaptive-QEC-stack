"""
Syndrome extraction pipeline.

Pipeline:
    raw measurement → stabilizer outcomes → detection events →
    detector graph → syndrome tensor

Data representation:
    S ∈ {0,1}^{shots × R × N_d}
    where R = QEC rounds, N_d = detector count per round

This becomes the common input format for every decoder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import stim

from adaptive_qec.data.models import DetectorRecord

logger = logging.getLogger(__name__)


@dataclass
class DetectorGraph:
    """
    Graph representation of detector relationships.

    Edges represent error mechanisms that can flip pairs of detectors.
    """
    num_detectors: int
    num_observables: int
    edges: list[tuple[int, int, float]]  # (det_i, det_j, probability)
    boundary_edges: list[tuple[int, float]]  # (det_i, probability)
    coordinates: Optional[np.ndarray] = None  # (num_detectors, ndim)


class SyndromeExtractor:
    """
    Extracts detection events from raw QPU measurements.

    Supports two modes:
    1. Stim-native: Use Stim's compiled detector sampler (for benchmarks)
    2. Raw QPU: Convert raw measurement bitstrings to detection events
       using the circuit's detector definitions
    """

    def __init__(self, stim_circuit: stim.Circuit) -> None:
        self._circuit = stim_circuit
        self._num_detectors = stim_circuit.num_detectors
        self._num_observables = stim_circuit.num_observables
        self._num_measurements = stim_circuit.num_measurements

        # Pre-compile the detector sampler for benchmark mode
        self._compiled_sampler = None

        # Build detector-to-measurement mapping
        self._detector_specs = self._build_detector_specs()

    def _build_detector_specs(self) -> list[list[int]]:
        """
        Build mapping from each detector to the measurement indices
        it depends on.

        Each detector is a parity check on a subset of measurements.
        A detection event fires when that parity is 1 (odd).
        """
        specs: list[list[int]] = []
        measurement_offset = 0
        total_measurements = self._num_measurements
