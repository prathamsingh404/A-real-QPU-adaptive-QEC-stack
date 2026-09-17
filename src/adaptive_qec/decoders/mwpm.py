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
            batch_shots = end - start
            per_shot_times[start:end] = batch_time / batch_shots

        t_total = time.perf_counter() - t_start

        # Memory measurement
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Combine predictions
        all_preds = np.vstack(all_predictions).astype(np.uint8)

        # Compute logical errors
        # A logical error occurs when the predicted correction, combined
        # with the actual observable flip, results in a logical flip
        logical_errors = np.any(all_preds != observable_flips, axis=1)
        num_errors = int(logical_errors.sum())
        error_rate = num_errors / shots

        # Latency statistics (convert to microseconds)
        latency_us = per_shot_times * 1e6

        metrics = DecoderMetrics(
            total_shots=shots,
            num_logical_errors=num_errors,
            logical_error_rate=error_rate,
            decode_time_s=t_total,
            per_shot_latency_us=latency_us,
            latency_mean_us=float(latency_us.mean()),
            latency_p50_us=float(np.percentile(latency_us, 50)),
            latency_p95_us=float(np.percentile(latency_us, 95)),
            latency_p99_us=float(np.percentile(latency_us, 99)),
            latency_p999_us=float(np.percentile(latency_us, 99.9)),
            throughput_shots_per_s=shots / t_total if t_total > 0 else 0,
            peak_memory_mb=peak / (1024 * 1024),
        )

        logger.info(
            f"MWPM decode: {shots} shots, "
            f"LER={error_rate:.6f} ({num_errors}/{shots}), "
            f"time={t_total:.3f}s, "
            f"throughput={metrics.throughput_shots_per_s:.0f} shots/s, "
            f"P99={metrics.latency_p99_us:.1f}μs"
        )
        return metrics

    def decode_burst_aware(
        self,
        syndromes: np.ndarray,
        observable_flips: np.ndarray,
        num_rounds: int,
        num_detectors_per_round: int,
        burst_detector: Optional[Any] = None,
    ) -> tuple[DecoderMetrics, DecoderMetrics, dict[str, Any]]:
        """
        Compare standard MWPM vs burst-aware MWPM on identical syndrome data.

        Burst-aware strategy:
        1. Identify shots and rounds with correlated error bursts.
        2. Mitigate anomalous burst detector clusters that violate independent error models.
        3. Decode with and without burst mitigation and return comparative metrics.

        Returns:
            (standard_metrics, burst_aware_metrics, burst_summary)
        """
        from adaptive_qec.noise.burst_detector import BurstDetector, reshape_syndromes_to_tensor

        detector = burst_detector or BurstDetector()
        shots = syndromes.shape[0]

        # 1. Standard decode
        standard_metrics = self.decode_batch(syndromes, observable_flips)

        # 2. Burst-aware preprocessing
        mitigated_syndromes = syndromes.copy().astype(np.uint8)
        burst_shot_count = 0
        total_bursts_found = 0

        for i in range(shots):
            shot_tensor = reshape_syndromes_to_tensor(
                syndromes[i], num_rounds, num_detectors_per_round
            )
            analysis = detector.analyze(shot_tensor, num_detectors_per_round)
            if analysis.bursts_detected:
