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
