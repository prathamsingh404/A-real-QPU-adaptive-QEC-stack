"""Tests for Union-Find decoder.

Validates correctness, performance, and comparison against MWPM baseline.
"""

import numpy as np
import pytest
import stim

from adaptive_qec.config import NoiseConfig
from adaptive_qec.decoders.base import DecoderMetrics
from adaptive_qec.decoders.mwpm import MWPMDecoder
from adaptive_qec.decoders.union_find import (
    DetectorGraph,
    UnionFindDecoder,
    UnionFindForest,
    build_detector_graph,
)
from adaptive_qec.decoders.registry import get_decoder
from adaptive_qec.qec.codes import create_code


# ---------------------------------------------------------------------------
# Unit tests for UnionFindForest
# ---------------------------------------------------------------------------

class TestUnionFindForest:
