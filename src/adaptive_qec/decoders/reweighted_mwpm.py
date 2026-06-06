"""
Online Reweighted MWPM Decoder (B3: Bhardwaj et al., PRX Quantum 2026 / DGR style).

Implements sliding-window syndrome-based dynamic edge reweighting for MWPM:
    1. Ingests syndromes into a rolling observation window.
    2. Computes empirical detector defect and pairwise coincidence rates.
    3. Smooths empirical estimates with base detector error model priors via EWMA:
           p_hat = (1 - alpha) * p_base + alpha * p_empirical
           w_e = ln((1 - p_hat) / p_hat)
    4. Rebuilds the PyMatching matching graph periodically at block intervals.
    5. Decodes subsequent shots with the updated matching graph.

References:
    - Bhardwaj et al., "Real-time adaptive decoding of quantum error-correcting codes"
      PRX Quantum (2026).
    - Spitz et al., "Adaptive weight estimator for quantum error correction in a
      time-dependent environment" (arXiv 2023).
"""

from __future__ import annotations

import collections
import logging
import time
import tracemalloc
from typing import Any, Optional

import numpy as np
import pymatching
import stim

from adaptive_qec.decoders.base import Correction, Decoder, DecoderMetrics
from adaptive_qec.decoders.dem_calibration import CalibratedDEM, DEMCalibrator

logger = logging.getLogger(__name__)


class OnlineReweightedMWPM(Decoder):
    """
    Online Reweighted MWPM Decoder (Baseline B3).

    Maintains a sliding window of recent syndromes, periodically re-estimates
    edge error probabilities from syndrome statistics without oracle access,
    and updates the matching graph.
    """

    def __init__(
        self,
        window_size: int = 200,
        update_interval: int = 50,
        smoothing: float = 0.35,
        min_probability: float = 1e-6,
        max_probability: float = 0.499,
    ) -> None:
        self.window_size = window_size
        self.update_interval = update_interval
        self.smoothing = smoothing
        self.min_probability = min_probability
        self.max_probability = max_probability

        self._calibrator: Optional[DEMCalibrator] = None
        self._current_matching: Optional[pymatching.Matching] = None
        self._circuit: Optional[stim.Circuit] = None
        self._base_dem: Optional[stim.DetectorErrorModel] = None

        self._num_detectors: int = 0
        self._num_observables: int = 0
        self._syndrome_buffer: collections.deque = collections.deque(maxlen=window_size)
        self._shots_since_update: int = 0
        self.total_reweights: int = 0

    @property
    def name(self) -> str:
        return "reweighted_mwpm"

    def configure(self, **kwargs: Any) -> None:
        """Configure decoder with circuit or DEM."""
        circuit = kwargs.get("circuit")
        dem = kwargs.get("dem")

        if circuit is None and dem is None:
            raise ValueError("Must provide 'circuit' or 'dem' to configure OnlineReweightedMWPM")

        if circuit is not None:
            self._circuit = circuit
            self._calibrator = DEMCalibrator(
                circuit=circuit,
                min_probability=self.min_probability,
                max_probability=self.max_probability,
            )
            dem = circuit.detector_error_model(decompose_errors=True)
        else:
            self._base_dem = dem
            # For standalone DEM, construct calibrator with fallback surface circuit or internal edges
            self._calibrator = DEMCalibrator(
                min_probability=self.min_probability,
                max_probability=self.max_probability,
            )

        self._base_dem = dem
        self._num_detectors = dem.num_detectors
        self._num_observables = dem.num_observables
        self._current_matching = pymatching.Matching.from_detector_error_model(dem)
        self._syndrome_buffer.clear()
        self._shots_since_update = 0
        self.total_reweights = 0

    def update_weights(self, window_syndromes: np.ndarray) -> None:
        """Trigger graph reweighting from an explicit syndrome window."""
        if self._calibrator is None:
            return
        calibrated: CalibratedDEM = self._calibrator.calibrate_from_syndromes(
            syndromes=window_syndromes,
            smoothing=self.smoothing,
        )
        self._current_matching = calibrated.to_matching()
        self.total_reweights += 1

    def decode(self, syndrome: np.ndarray) -> Correction:
        """Decode single shot or batch while streaming into the sliding window."""
        if self._current_matching is None:
            raise RuntimeError("Decoder not configured. Call configure() first.")

        syndrome = np.asarray(syndrome, dtype=np.uint8)

        if syndrome.ndim == 1:
            self._syndrome_buffer.append(syndrome)
            self._shots_since_update += 1

            if self._shots_since_update >= self.update_interval and len(self._syndrome_buffer) >= 20:
                window_arr = np.array(self._syndrome_buffer)
                self.update_weights(window_arr)
                self._shots_since_update = 0
