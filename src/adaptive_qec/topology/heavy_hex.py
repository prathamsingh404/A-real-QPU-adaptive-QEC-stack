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
                    edges.append(list(qargs))
            num_qubits = target.num_qubits
            return cls.from_coupling_map(edges, num_qubits)
        except AttributeError:
            # Fallback: try coupling_map attribute
            cm = backend.configuration().coupling_map
            n = backend.configuration().n_qubits
            return cls.from_coupling_map(cm, n)

    @classmethod
    def synthetic(cls, rows: int = 5, cols: int = 5) -> HeavyHexTopology:
        """
        Generate a synthetic heavy-hex topology.

        Creates a heavy-hex lattice with the specified grid dimensions.
        Each hex cell has 6 vertices + flag qubits on edges.
        """
        topo = cls()
        edges = []
        qubit_id = 0

        # Simple heavy-hex: alternating rows of degree-2 and degree-3 qubits
        # Grid-based generation
        grid = {}
        for r in range(rows):
            for c in range(cols):
                grid[(r, c)] = qubit_id
                qubit_id += 1

        topo.num_qubits = qubit_id

        for r in range(rows):
            for c in range(cols):
                q = grid[(r, c)]
                # Horizontal neighbors
                if c + 1 < cols:
                    edges.append((q, grid[(r, c + 1)]))
                # Vertical neighbors (heavy-hex / brickwall alternating pattern:
                # even rows connect down at c % 4 == 0, odd rows connect down at c % 4 == 2)
                connect_down = (r % 2 == 0 and c % 4 == 0) or (r % 2 == 1 and c % 4 == 2)
                if r + 1 < rows and connect_down:
                    edges.append((q, grid[(r + 1, c)]))

        topo.edges = edges
        for q1, q2 in edges:
            topo.adjacency.setdefault(q1, set()).add(q2)
            topo.adjacency.setdefault(q2, set()).add(q1)
        for q in range(topo.num_qubits):
            topo.adjacency.setdefault(q, set())
        topo._degree = {q: len(n) for q, n in topo.adjacency.items()}

        return topo

    def degree(self, qubit: int) -> int:
        """Get the degree (number of neighbors) of a qubit."""
        return self._degree.get(qubit, 0)

    def neighbors(self, qubit: int) -> set[int]:
        """Get the neighbors of a qubit."""
        return self.adjacency.get(qubit, set())

    def compute_metrics(self) -> TopologyMetrics:
        """Compute topology metrics."""
        degrees = list(self._degree.values())
        if not degrees:
            return TopologyMetrics(
                num_qubits=0, num_edges=0, min_degree=0, max_degree=0,
                avg_degree=0.0, degree_distribution={}, diameter=0,
                is_heavy_hex=False,
            )

        # Degree distribution
        deg_dist: dict[int, int] = {}
        for d in degrees:
            deg_dist[d] = deg_dist.get(d, 0) + 1

        # Diameter via BFS from each node (for small topologies)
        diameter = 0
        if self.num_qubits <= 500:
            for start in range(min(self.num_qubits, 20)):
                dist = self._bfs_distances(start)
                max_dist = max(dist.values()) if dist else 0
                diameter = max(diameter, max_dist)
        else:
            # Sample a few nodes for large topologies
            sample = np.random.choice(self.num_qubits, min(10, self.num_qubits), replace=False)
            for start in sample:
                dist = self._bfs_distances(int(start))
                max_dist = max(dist.values()) if dist else 0
                diameter = max(diameter, max_dist)

        # Heavy-hex detection: max degree ≤ 3, most qubits degree 2
        max_deg = max(degrees)
        is_heavy_hex = (max_deg <= 3 and deg_dist.get(2, 0) > len(degrees) * 0.3)

        return TopologyMetrics(
            num_qubits=self.num_qubits,
            num_edges=len(self.edges),
            min_degree=min(degrees),
            max_degree=max_deg,
            avg_degree=float(np.mean(degrees)),
            degree_distribution=deg_dist,
            diameter=diameter,
            is_heavy_hex=is_heavy_hex,
        )

    def _bfs_distances(self, start: int) -> dict[int, int]:
        """BFS from start, returns distances to all reachable nodes."""
        from collections import deque
        dist = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in self.adjacency.get(node, set()):
                if neighbor not in dist:
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)
        return dist

    def find_shortest_path(self, start: int, end: int) -> list[int]:
        """Find shortest path between two qubits."""
        from collections import deque
        if start == end:
            return [start]

        visited = {start: None}
        queue = deque([start])

        while queue:
            node = queue.popleft()
            for neighbor in self.adjacency.get(node, set()):
                if neighbor not in visited:
                    visited[neighbor] = node
                    if neighbor == end:
                        # Reconstruct path
                        path = [end]
                        current = end
                        while visited[current] is not None:
                            current = visited[current]
                            path.append(current)
                        return list(reversed(path))
                    queue.append(neighbor)

        return []  # No path found

    def swap_distance(self, q1: int, q2: int) -> int:
        """
        Compute the SWAP distance between two qubits.

        SWAP distance = shortest path length - 1
        (number of SWAP gates needed to make q1 and q2 adjacent)
        """
        path = self.find_shortest_path(q1, q2)
        return max(0, len(path) - 2)  # -2 because the path includes both endpoints
