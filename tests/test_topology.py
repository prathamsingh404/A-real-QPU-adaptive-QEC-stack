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

    def test_from_coupling_map(self):
        # A simple hex cell: 0-1-2-3-4-5-0
        edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]
        topo = HeavyHexTopology.from_coupling_map(edges, num_qubits=6)

        assert topo.num_qubits == 6
        assert len(topo.edges) == 6
        assert topo.degree(0) == 2
        assert topo.neighbors(0) == {1, 5}

    def test_synthetic_heavy_hex(self):
        topo = HeavyHexTopology.synthetic(rows=6, cols=6)
        assert topo.num_qubits == 36
        metrics = topo.compute_metrics()

        assert isinstance(metrics, TopologyMetrics)
        assert metrics.num_qubits == 36
        assert metrics.num_edges > 0
        assert metrics.max_degree <= 3
        assert metrics.diameter > 0

    def test_shortest_path_and_swap_distance(self):
        edges = [[0, 1], [1, 2], [2, 3], [3, 4]]
        topo = HeavyHexTopology.from_coupling_map(edges, num_qubits=5)

        path = topo.find_shortest_path(0, 3)
        assert path == [0, 1, 2, 3]

        # Adjacent qubits require 0 SWAPs
        assert topo.swap_distance(0, 1) == 0
