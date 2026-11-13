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
        # 0 to 2 requires 1 SWAP (len(path) - 2 = 3 - 2 = 1)
        assert topo.swap_distance(0, 2) == 1
        # 0 to 3 requires 2 SWAPs
        assert topo.swap_distance(0, 3) == 2


class TestEmbeddingFinder:
    """Tests for SurfaceCode EmbeddingFinder."""

    def test_find_embedding_d3(self):
        # Synthetic heavy-hex with plenty of qubits
        topo = HeavyHexTopology.synthetic(rows=7, cols=7)
        finder = EmbeddingFinder(topo)

        embedding = finder.find_embedding(distance=3)
        assert isinstance(embedding, SurfaceCodeEmbedding)
        assert embedding.distance == 3
        # d=3 surface code requires 9 data qubits
        assert len(embedding.data_map) == 9
        # ancillas mapped
        assert len(embedding.ancilla_map) > 0

        qubits_used = embedding.all_physical_qubits()
        assert len(qubits_used) > 9
        assert embedding.circuit_depth_overhead() >= 1.0

        d_dict = embedding.to_dict()
        assert d_dict["distance"] == 3
        assert d_dict["num_data_qubits"] == 9

    def test_surface_code_circuit_with_embedding(self):
        topo = HeavyHexTopology.synthetic(rows=6, cols=6)
        finder = EmbeddingFinder(topo)
        embedding = finder.find_embedding(distance=3)

        code = create_code("surface", distance=3, rounds=3)
        noise = NoiseConfig()
        noise.gate.two_qubit = 0.005

        circuit = code.generate_circuit(noise=noise, embedding=embedding)
        assert isinstance(circuit, stim.Circuit)
        assert circuit.num_detectors > 0
        assert circuit.num_observables == 1

    def test_heavy_hex_heron_scale_layout(self):
        """Heron-scale heavy-hex topology (11x11 grid) satisfies degree <= 3 constraint."""
        topo = HeavyHexTopology.synthetic(rows=11, cols=11)
        assert topo.num_qubits == 121
        metrics = topo.compute_metrics()
        assert metrics.max_degree <= 3
        assert 1.5 <= metrics.avg_degree <= 2.8
        assert metrics.diameter > 0

    def test_disconnected_qubits_shortest_path(self):
        """Shortest path between disconnected components returns empty path."""
        topo = HeavyHexTopology.from_coupling_map([[0, 1], [2, 3]], num_qubits=4)
        path = topo.find_shortest_path(0, 3)
        assert path == []

    def test_embedding_d3_vs_d5_scaling(self):
        """Distance 5 embedding maps strictly more data qubits than distance 3."""
        topo = HeavyHexTopology.synthetic(rows=11, cols=11)
        finder = EmbeddingFinder(topo)

        emb3 = finder.find_embedding(distance=3)
        emb5 = finder.find_embedding(distance=5)

        assert len(emb3.data_map) == 9
        assert len(emb5.data_map) == 25
        assert len(emb5.all_physical_qubits()) > len(emb3.all_physical_qubits())

