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
