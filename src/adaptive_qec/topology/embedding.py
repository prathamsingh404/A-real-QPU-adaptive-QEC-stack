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
