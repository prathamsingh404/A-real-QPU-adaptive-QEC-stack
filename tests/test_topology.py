"""Tests for Heavy-Hex Topology and Surface Code Embedding (Problem 1).

Validates coupling map parsing, heavy-hex graph representation, BFS shortest path,
SWAP distance computation, greedy surface code embedding search, and circuit generation.
"""

import pytest
import stim

from adaptive_qec.config import NoiseConfig
from adaptive_qec.qec.codes import create_code
from adaptive_qec.topology.embedding import EmbeddingFinder, SurfaceCodeEmbedding
from adaptive_qec.topology.heavy_hex import HeavyHexTopology, TopologyMetrics


class TestHeavyHexTopology:
    """Tests for HeavyHexTopology class."""

