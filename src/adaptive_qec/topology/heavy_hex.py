"""
Heavy-hex topology analysis for IBM quantum processors.

IBM's Heron r2 (ibm_marrakesh, 156 qubits) uses a heavy-hex lattice,
not the square grid that surface codes naturally sit on. This mismatch
is the primary architectural barrier for IBM QEC scaling.

This module:
    1. Parses coupling maps from IBM backends
    2. Builds a graph representation of the heavy-hex topology
    3. Identifies heavy-hex unit cells and connectivity structure
    4. Computes topology metrics (degree distribution, diameter, etc.)

Sources:
    - IBM heavy-hex lattice architecture documentation
    - Chamberland et al., "Topological and Subsystem Codes on Low-Degree
      Graphs with Flag Qubits" (2020)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TopologyMetrics:
    """Metrics characterizing a qubit topology."""
    num_qubits: int
    num_edges: int
    min_degree: int
    max_degree: int
    avg_degree: float
    degree_distribution: dict[int, int]  # degree → count
    diameter: int  # longest shortest path
    is_heavy_hex: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_qubits": self.num_qubits,
            "num_edges": self.num_edges,
            "min_degree": self.min_degree,
            "max_degree": self.max_degree,
            "avg_degree": round(self.avg_degree, 3),
            "degree_distribution": self.degree_distribution,
            "diameter": self.diameter,
            "is_heavy_hex": self.is_heavy_hex,
        }


class HeavyHexTopology:
    """
    Representation and analysis of IBM's heavy-hex qubit topology.

    Heavy-hex is a modified hexagonal lattice where each edge of the
    hex lattice has a "flag" qubit inserted, reducing the maximum degree
    from 3 to 2-3 while increasing total qubit count.

    This class can be initialized either from:
        - A coupling map (list of [q1, q2] pairs)
        - An IBM backend object
        - A synthetic heavy-hex of specified size
    """

    def __init__(self) -> None:
        self.num_qubits: int = 0
        self.edges: list[tuple[int, int]] = []
        self.adjacency: dict[int, set[int]] = {}
        self._degree: dict[int, int] = {}

    @classmethod
    def from_coupling_map(cls, coupling_map: list[list[int]], num_qubits: int) -> HeavyHexTopology:
        """Build from a coupling map (list of [q1, q2] pairs)."""
        topo = cls()
        topo.num_qubits = num_qubits

        seen_edges: set[tuple[int, int]] = set()
        for pair in coupling_map:
            q1, q2 = min(pair), max(pair)
            if (q1, q2) not in seen_edges:
                seen_edges.add((q1, q2))
                topo.edges.append((q1, q2))

        # Build adjacency
        for q1, q2 in topo.edges:
            topo.adjacency.setdefault(q1, set()).add(q2)
            topo.adjacency.setdefault(q2, set()).add(q1)

        # Ensure all qubits are in adjacency
        for q in range(num_qubits):
            topo.adjacency.setdefault(q, set())

        topo._degree = {q: len(neighbors) for q, neighbors in topo.adjacency.items()}
        return topo

    @classmethod
    def from_backend(cls, backend: Any) -> HeavyHexTopology:
        """Build from an IBM backend object."""
        try:
            target = backend.target
            coupling_map = list(target.operation_names_for_qargs)
            # Extract from target's two-qubit gate entries
            edges = []
            for qargs in target.qargs:
                if len(qargs) == 2:
