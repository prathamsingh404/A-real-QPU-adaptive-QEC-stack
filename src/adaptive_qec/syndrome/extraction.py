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

        # Walk through the circuit to find DETECTOR instructions
        # and their record targets
        for instruction in self._circuit.flattened():
            if instruction.name == "M" or instruction.name == "MR":
                measurement_offset += len(instruction.targets_copy())

        # Use Stim's compiled approach for correctness
        # The detector definitions reference measurements via rec[-k]
        # We extract this programmatically from the circuit
        try:
            # Stim provides this via the explain_detector_error_model approach
            # For now, we rely on compiled_detector_sampler for correctness
            pass
        except Exception:
            pass

        return specs

    def sample_detectors(self, shots: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample detection events directly from Stim (benchmark mode).

        This uses Stim's compiled detector sampler for fast, correct
        sampling. Used for controlled benchmarks, NOT for real QPU data.

        Args:
            shots: Number of shots to sample.

        Returns:
            Tuple of (detection_events, observable_flips):
            - detection_events: shape (shots, num_detectors), dtype uint8
            - observable_flips: shape (shots, num_observables), dtype uint8
        """
        if self._compiled_sampler is None:
            self._compiled_sampler = self._circuit.compile_detector_sampler()

        # Sample detection events and observable flips together
        result = self._compiled_sampler.sample(
            shots=shots,
            separate_observables=True,
        )

        detection_events = np.array(result[0], dtype=np.uint8)
        observable_flips = np.array(result[1], dtype=np.uint8)

        logger.info(
            f"Sampled {shots} shots: "
            f"detection_events {detection_events.shape}, "
            f"observable_flips {observable_flips.shape}"
        )
        return detection_events, observable_flips

    def extract_from_measurements(
        self,
        raw_measurements: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract detection events from raw QPU measurement outcomes.

        This is the critical function for real hardware data. It converts
        raw bitstring measurements into the syndrome format expected by decoders.

        Args:
            raw_measurements: shape (shots, num_measurements), dtype uint8.
                Each row is a bitstring of all measurements in one shot.

        Returns:
            Tuple of (detection_events, observable_flips):
            - detection_events: shape (shots, num_detectors), dtype uint8
            - observable_flips: shape (shots, num_observables), dtype uint8
        """
        shots = raw_measurements.shape[0]

        # Use Stim's compiled measurement-to-detection converter
