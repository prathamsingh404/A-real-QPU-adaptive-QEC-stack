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
