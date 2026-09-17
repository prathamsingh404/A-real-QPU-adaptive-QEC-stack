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


