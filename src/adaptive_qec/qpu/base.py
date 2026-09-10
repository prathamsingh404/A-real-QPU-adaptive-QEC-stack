"""
Abstract QPU backend interface.

This is the non-negotiable abstraction that decouples research code
from any single provider's API. All QPU interactions go through this
interface.

    qpu.run(circuit, shots)
    qpu.get_calibration()
    qpu.get_topology()
    qpu.get_backend_info()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class BackendInfo:
    """Static information about a QPU backend."""
    name: str
    provider: str
    num_qubits: int
    topology_type: str
    version: str
    status: str  # "online" | "offline" | "maintenance"
    max_shots: int
    max_circuits: int
    basis_gates: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologyInfo:
    """Qubit connectivity information."""
    num_qubits: int
    coupling_map: list[tuple[int, int]]
    adjacency: dict[int, list[int]] = field(default_factory=dict)

    def neighbors(self, qubit: int) -> list[int]:
        """Get neighboring qubits."""
        if not self.adjacency:
            self._build_adjacency()
        return self.adjacency.get(qubit, [])

    def _build_adjacency(self) -> None:
        """Build adjacency dict from coupling map."""
        self.adjacency = {}
        for q1, q2 in self.coupling_map:
            self.adjacency.setdefault(q1, []).append(q2)
            self.adjacency.setdefault(q2, []).append(q1)


@dataclass
class QubitCalibration:
    """Calibration data for a single qubit."""
    qubit_index: int
    t1_us: Optional[float] = None            # T1 in microseconds
    t2_us: Optional[float] = None            # T2 in microseconds
    readout_error: Optional[float] = None    # readout assignment error
    readout_length_ns: Optional[float] = None
    single_qubit_gate_error: Optional[float] = None
    frequency_ghz: Optional[float] = None
    anharmonicity_ghz: Optional[float] = None


@dataclass
class GateCalibration:
