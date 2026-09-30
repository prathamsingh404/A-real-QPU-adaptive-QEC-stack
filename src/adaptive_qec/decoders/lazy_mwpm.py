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

@property
    def name(self) -> str:
        return "lazy_mwpm"

def configure(self, **kwargs: Any) -> None:
        self._mwpm.configure(**kwargs)
        self._uf.configure(**kwargs)
        self._num_detectors = self._mwpm._num_detectors
        self._num_observables = self._mwpm._num_observables

def is_ambiguous(self, syndrome: np.ndarray) -> bool:
        return bool(syndrome.sum() > self.defect_threshold)

def decode(self, syndrome: np.ndarray) -> Correction:
        if syndrome.ndim == 1:
            if self.is_ambiguous(syndrome):
                return self._mwpm.decode(syndrome)
            return self._uf.decode(syndrome)

defect_counts = syndrome.sum(axis=1)
        mwpm_mask = defect_counts > self.defect_threshold
        uf_mask = ~mwpm_mask
        predictions = np.zeros((syndrome.shape[0], self._num_observables), dtype=np.uint8)
        if np.any(mwpm_mask):
            predictions[mwpm_mask] = self._mwpm.decode(syndrome[mwpm_mask]).observable_corrections
        if np.any(uf_mask):
            predictions[uf_mask] = self._uf.decode(syndrome[uf_mask]).observable_corrections
        return Correction(observable_corrections=predictions)
