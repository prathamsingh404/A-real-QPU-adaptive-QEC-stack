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
    """Test the core union-find data structure."""

    def test_find_self(self):
        uf = UnionFindForest(5)
        for i in range(5):
            assert uf.find(i) == i

    def test_union_and_find(self):
        uf = UnionFindForest(5)
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)

    def test_transitive_union(self):
        uf = UnionFindForest(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(2)

    def test_parity_tracking(self):
        uf = UnionFindForest(5)
        uf.parity[0] = 1  # one defect
        uf.parity[1] = 1  # one defect
        uf.union(0, 1)
        root = uf.find(0)
        assert uf.parity[root] == 2  # even → even cluster

    def test_boundary_connection(self):
        uf = UnionFindForest(5)
        uf.boundary_connected[3] = True
        uf.parity[0] = 1  # odd parity
        uf.union(0, 3)
        assert uf.is_even(0)  # boundary absorbs odd parity

    def test_is_even_zero_parity(self):
        uf = UnionFindForest(3)
        assert uf.is_even(0)  # zero defects = even

    def test_is_even_odd_parity(self):
        uf = UnionFindForest(3)
        uf.parity[0] = 1
        assert not uf.is_even(0)

    def test_path_compression(self):
        """Path compression should flatten chains."""
        uf = UnionFindForest(10)
