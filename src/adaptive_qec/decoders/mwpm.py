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
