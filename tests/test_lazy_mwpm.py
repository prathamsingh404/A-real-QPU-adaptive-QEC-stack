import pytest
import numpy as np
import stim
from adaptive_qec.decoders.lazy_mwpm import LazyMWPMDecoder
from adaptive_qec.decoders.registry import get_decoder

def test_lazy_mwpm_initialization():
    dec = LazyMWPMDecoder(defect_threshold=10)
    assert dec.name == 'lazy_mwpm'
    assert dec.defect_threshold == 10
