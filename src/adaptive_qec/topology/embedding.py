"""
Surface code embedding on hardware topologies.

Maps logical surface code qubits (data + ancilla) onto physical hardware
qubits, accounting for connectivity constraints.

The key challenge: surface codes require a square grid, but IBM's heavy-hex
lattice has max degree 3 and irregular connectivity. Embedding requires
SWAP routing, which introduces idle time and noise overhead.

Embedding quality metrics:
    - SWAP count: total SWAPs needed per QEC round
    - Circuit depth overhead: increase in circuit depth vs ideal
    - Idle time: total idle slots where qubits accumulate noise
    - Connectivity deficit: fraction of required edges missing

Sources:
    - IBM/ETH Zurich "fold-unfold" embedding strategy
    - Lao & Almudever, "Mapping of lattice surgery-based quantum circuits
      on surface code architectures" (2019)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np

from adaptive_qec.topology.heavy_hex import HeavyHexTopology

logger = logging.getLogger(__name__)


class CodeEmbedding(Protocol):
    """Protocol for code-to-hardware qubit mappings."""

    def data_qubit_map(self) -> dict[tuple[int, int], int]:
        """Map logical (row, col) → physical qubit index for data qubits."""
        ...

    def ancilla_qubit_map(self) -> dict[tuple[int, int], int]:
        """Map logical (row, col) → physical qubit index for ancilla qubits."""
        ...

    def swap_count(self) -> int:
        """Total SWAPs needed per QEC round."""
        ...

    def circuit_depth_overhead(self) -> float:
        """Multiplicative depth overhead vs ideal square grid."""
        ...


@dataclass
class EmbeddingScore:
    """Quality score for an embedding."""
    swap_count: int
    circuit_depth_overhead: float
    idle_time_slots: int
    connectivity_deficit: float  # 0.0 = perfect, 1.0 = no connectivity
    total_score: float  # Combined weighted score (lower = better)

    def to_dict(self) -> dict[str, Any]:
        return {
            "swap_count": self.swap_count,
            "circuit_depth_overhead": round(self.circuit_depth_overhead, 3),
            "idle_time_slots": self.idle_time_slots,
            "connectivity_deficit": round(self.connectivity_deficit, 4),
            "total_score": round(self.total_score, 4),
        }


@dataclass
class SurfaceCodeEmbedding:
    """
    A specific embedding of a surface code on a hardware topology.

    Maps logical surface code layout to physical qubit indices,
    with computed routing overhead.
    """
    distance: int
    data_map: dict[tuple[int, int], int] = field(default_factory=dict)
    ancilla_map: dict[tuple[int, int], int] = field(default_factory=dict)
    _swap_count: int = 0
    _depth_overhead: float = 1.0
    _idle_slots: int = 0

    def data_qubit_map(self) -> dict[tuple[int, int], int]:
        return self.data_map

    def ancilla_qubit_map(self) -> dict[tuple[int, int], int]:
        return self.ancilla_map

    def swap_count(self) -> int:
        return self._swap_count

    def circuit_depth_overhead(self) -> float:
        return self._depth_overhead

    def all_physical_qubits(self) -> set[int]:
        """All physical qubits used by this embedding."""
        return set(self.data_map.values()) | set(self.ancilla_map.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "distance": self.distance,
            "num_data_qubits": len(self.data_map),
            "num_ancilla_qubits": len(self.ancilla_map),
            "total_physical_qubits": len(self.all_physical_qubits()),
            "swap_count": self._swap_count,
            "depth_overhead": round(self._depth_overhead, 3),
            "idle_slots": self._idle_slots,
        }


class EmbeddingFinder:
    """
    Find optimal surface code embeddings on hardware topologies.

    Search strategy:
        1. Generate candidate starting positions
        2. For each start, greedily assign data qubits using BFS
        3. Assign ancilla qubits to remaining nearby physical qubits
        4. Score each embedding by SWAP count + idle time
        5. Return the best scoring embedding
    """

    def __init__(self, topology: HeavyHexTopology) -> None:
        self.topology = topology

    def find_embedding(
        self,
        distance: int,
        max_candidates: int = 20,
    ) -> SurfaceCodeEmbedding:
        """
        Find the best embedding for a surface code of given distance.

        Args:
            distance: Code distance (3, 5, 7, ...)
            max_candidates: Maximum starting positions to try.

        Returns:
            Best SurfaceCodeEmbedding found.
        """
        n_data = distance ** 2
        n_ancilla = (distance - 1) ** 2 + (distance - 1) ** 2
        total_needed = n_data + n_ancilla

        if total_needed > self.topology.num_qubits:
            logger.warning(
                f"Need {total_needed} qubits for d={distance} but topology "
                f"has only {self.topology.num_qubits}"
            )

        # Generate candidate starting positions
        # Prefer high-connectivity qubits as starting points
        candidates = sorted(
            range(self.topology.num_qubits),
            key=lambda q: -self.topology.degree(q),
        )[:max_candidates]

        best_embedding: Optional[SurfaceCodeEmbedding] = None
        best_score = float("inf")

        for start_qubit in candidates:
            embedding = self._try_embedding(distance, start_qubit)
            if embedding is None:
                continue

            score = self._score_embedding(embedding)
            if score.total_score < best_score:
                best_score = score.total_score
                best_embedding = embedding

        if best_embedding is None:
            logger.warning(f"No valid embedding found for d={distance}")
            return SurfaceCodeEmbedding(distance=distance)

        logger.info(
            f"Best embedding for d={distance}: "
            f"{len(best_embedding.all_physical_qubits())} physical qubits, "
            f"score={best_score:.2f}"
        )
        return best_embedding

    def _try_embedding(
        self, distance: int, start_qubit: int
    ) -> Optional[SurfaceCodeEmbedding]:
        """Try to embed a surface code starting from a specific qubit."""
        from collections import deque

        n_data = distance ** 2
        embedding = SurfaceCodeEmbedding(distance=distance)

        # BFS from start to claim physical qubits for data qubits
        used_physical: set[int] = set()
        physical_queue = deque([start_qubit])
        data_positions: list[int] = []

        while physical_queue and len(data_positions) < n_data:
            pq = physical_queue.popleft()
            if pq in used_physical:
                continue
            used_physical.add(pq)
            data_positions.append(pq)

            for neighbor in sorted(self.topology.neighbors(pq)):
                if neighbor not in used_physical:
                    physical_queue.append(neighbor)

        if len(data_positions) < n_data:
            return None  # Not enough connected qubits

        # Assign data qubits to logical grid positions
        for idx, pq in enumerate(data_positions):
            row = idx // distance
            col = idx % distance
            embedding.data_map[(row, col)] = pq

        # Find ancilla positions: neighbors of data qubits not yet used
        ancilla_positions: list[int] = []
        n_ancilla_needed = 2 * (distance - 1) * distance  # X and Z stabilizers

        for pq in data_positions:
            for neighbor in self.topology.neighbors(pq):
                if neighbor not in used_physical and neighbor not in ancilla_positions:
                    ancilla_positions.append(neighbor)
                    if len(ancilla_positions) >= n_ancilla_needed:
                        break
            if len(ancilla_positions) >= n_ancilla_needed:
                break

        for idx, pq in enumerate(ancilla_positions):
            row = idx // (distance - 1) if distance > 1 else 0
            col = idx % (distance - 1) if distance > 1 else 0
            embedding.ancilla_map[(row, col)] = pq

        # Compute SWAP overhead
        swap_count = 0
        for (r, c), pq in embedding.data_map.items():
            # Check connectivity to adjacent data qubits
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in embedding.data_map:
                    npq = embedding.data_map[(nr, nc)]
                    if npq not in self.topology.neighbors(pq):
                        swap_count += self.topology.swap_distance(pq, npq)

        embedding._swap_count = swap_count
        embedding._depth_overhead = 1.0 + (swap_count * 3) / max(distance * 4, 1)
