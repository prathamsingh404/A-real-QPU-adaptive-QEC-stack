"""
Lazy MWPM Hybrid Decoder Architecture
"""
from __future__ import annotations
import logging
import time
from typing import Any
import numpy as np
from adaptive_qec.decoders.base import Correction, Decoder, DecoderMetrics
from adaptive_qec.decoders.mwpm import MWPMDecoder
from adaptive_qec.decoders.union_find import UnionFindDecoder

logger = logging.getLogger(__name__)

class LazyMWPMDecoder(Decoder):
    """Hybrid decoder that routes defect clusters to MWPM or Union-Find."""
    def __init__(self, defect_threshold: int = 15) -> None:
        self.defect_threshold = defect_threshold
        self._mwpm = MWPMDecoder()
        self._uf = UnionFindDecoder()
        self._num_detectors = 0
        self._num_observables = 0
