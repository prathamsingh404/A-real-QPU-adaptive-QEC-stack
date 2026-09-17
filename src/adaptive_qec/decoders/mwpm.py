"""
MWPM decoder using PyMatching.

This is the fundamental baseline decoder. Measures:
    - logical error rate
    - latency (per shot, P50, P95, P99, P999)
    - throughput (shots/second)
    - memory usage
    - scaling with distance
"""

from __future__ import annotations

import logging
import time
import tracemalloc
from typing import Any, Optional

import numpy as np
import pymatching
import stim

from adaptive_qec.decoders.base import Correction, Decoder, DecoderMetrics

logger = logging.getLogger(__name__)


class MWPMDecoder(Decoder):
    """
    Minimum Weight Perfect Matching decoder via PyMatching.

    Constructs a Matching object from a Stim DetectorErrorModel,
    then decodes syndromes to predicted observable corrections.
    """

    def __init__(self) -> None:
        self._matching: Optional[pymatching.Matching] = None
        self._num_detectors: int = 0
        self._num_observables: int = 0

    @property
    def name(self) -> str:
        return "mwpm"

    def configure(self, **kwargs: Any) -> None:
        """
        Configure with a DetectorErrorModel or Stim Circuit.

        Args:
            dem: stim.DetectorErrorModel
            circuit: stim.Circuit (will extract DEM automatically)
        """
        dem = kwargs.get("dem")
        circuit = kwargs.get("circuit")

        if dem is None and circuit is not None:
            dem = circuit.detector_error_model(decompose_errors=True)

        if dem is None:
            raise ValueError("Must provide 'dem' or 'circuit' to configure MWPM decoder")

        self._matching = pymatching.Matching.from_detector_error_model(dem)
        self._num_detectors = dem.num_detectors
        self._num_observables = dem.num_observables

        logger.info(
            f"MWPM decoder configured: {self._num_detectors} detectors, "
            f"{self._num_observables} observables"
        )

    def decode(self, syndrome: np.ndarray) -> Correction:
        """
        Decode a single syndrome or batch.

        Args:
            syndrome: shape (num_detectors,) or (batch, num_detectors)

        Returns:
            Correction with observable predictions.
        """
        if self._matching is None:
            raise RuntimeError("Decoder not configured. Call configure() first.")

        if syndrome.ndim == 1:
            prediction = self._matching.decode(syndrome.astype(np.uint8))
            return Correction(
                observable_corrections=prediction.astype(np.uint8),
            )

        # Batch decode
        predictions = self._matching.decode_batch(syndrome.astype(np.uint8))
        return Correction(
            observable_corrections=predictions.astype(np.uint8),
        )

    def decode_batch(
        self,
        syndromes: np.ndarray,
        observable_flips: np.ndarray,
    ) -> DecoderMetrics:
        """
        Decode a batch and compute comprehensive metrics.

        Args:
            syndromes: shape (shots, num_detectors)
            observable_flips: shape (shots, num_observables) — actual flips

        Returns:
            DecoderMetrics with error rate, latency distribution, throughput.
        """
        if self._matching is None:
            raise RuntimeError("Decoder not configured. Call configure() first.")

        shots = syndromes.shape[0]

        # Track memory
        tracemalloc.start()

        # Per-shot timing for latency distribution
        per_shot_times = np.zeros(shots)

        # Batch decode with timing
        t_start = time.perf_counter()

        # Decode in mini-batches for latency measurement
        batch_size = min(1000, shots)
        all_predictions = []

        for start in range(0, shots, batch_size):
            end = min(start + batch_size, shots)
            batch = syndromes[start:end].astype(np.uint8)

            t_batch_start = time.perf_counter()
            predictions = self._matching.decode_batch(batch)
            t_batch_end = time.perf_counter()

            all_predictions.append(predictions)

            # Approximate per-shot latency within batch
            batch_time = (t_batch_end - t_batch_start)
