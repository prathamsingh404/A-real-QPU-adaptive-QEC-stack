"""
Deterministic Quantum Teleportation via Real-Time Dynamic Feedforward.

Teleports arbitrary single-qubit quantum states across heavy-hex transit links
using mid-circuit Bell-state measurement, sub-microsecond on-chip classical
feedforward Pauli corrections, and idle dynamical decoupling.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

logger = logging.getLogger(__name__)

CLASSICAL_TELEPORTATION_BOUND = 2.0 / 3.0  # 66.67% max fidelity without quantum entanglement


@dataclass(frozen=True)
class TargetQuantumState:
    """Specification of target single-qubit state to teleport."""
    name: str
    theta: float
    phi: float
    ideal_bloch_vector: Tuple[float, float, float]


def get_cardinal_state(name: str) -> TargetQuantumState:
    """Return ideal parameters and Bloch vector for cardinal benchmark states."""
    name = name.lower().strip()
    if name in ("0", "|0>"):
        return TargetQuantumState("|0>", 0.0, 0.0, (0.0, 0.0, 1.0))
    elif name in ("1", "|1>"):
        return TargetQuantumState("|1>", math.pi, 0.0, (0.0, 0.0, -1.0))
    elif name in ("+", "|+>"):
        return TargetQuantumState("|+>", math.pi / 2.0, 0.0, (1.0, 0.0, 0.0))
    elif name in ("-", "|->"):
        return TargetQuantumState("|->", math.pi / 2.0, math.pi, (-1.0, 0.0, 0.0))
    elif name in ("+i", "|+i>"):
        return TargetQuantumState("|+i>", math.pi / 2.0, math.pi / 2.0, (0.0, 1.0, 0.0))
    elif name in ("-i", "|-i>"):
        return TargetQuantumState("|-i>", math.pi / 2.0, 3.0 * math.pi / 2.0, (0.0, -1.0, 0.0))
    else:
        raise ValueError(f"Unknown cardinal state: {name}")


class DeterministicTeleportation:
    """
    Deterministic Quantum Teleportation on transmon lattices.

    Compares three experimental regimes:
        1. 'dynamic': Real-time mid-circuit Bell measurement + on-chip feedforward.
        2. 'post_selected': Static circuit discarding (c0, c1) != (0, 0) (25% yield).
        3. 'swap_network': Static unitary teleportation via SWAP gates.
    """

    def __init__(
        self,
        target_state: TargetQuantumState,
        regime: str = "dynamic",
        apply_dd: bool = True,
    ) -> None:
        self.target_state = target_state
        self.regime = regime
        self.apply_dd = apply_dd

    def build_circuit(self, measurement_basis: str = "Z") -> QuantumCircuit:
        """
        Build teleportation circuit for a given measurement basis on destination qubit.

        Args:
            measurement_basis: 'X', 'Y', or 'Z' for full quantum state tomography.

        Architecture:
            - Source qubit:      q[0] (prepared in target state |psi>)
            - Transit qubit:     q[1] (entangled with destination)
            - Destination qubit: q[2] (receives teleported state)
            - Classical BSM:     c_bsm[0], c_bsm[1]
            - Destination out:   c_out[0]
        """
        basis = measurement_basis.upper().strip()
        if basis not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid measurement basis: {measurement_basis}")

        q = QuantumRegister(3, name="q")
        c_bsm = ClassicalRegister(2, name="c_bsm")
        c_out = ClassicalRegister(1, name="c_out")
        qc = QuantumCircuit(q, c_bsm, c_out, name=f"teleport_{self.target_state.name}_{basis}_{self.regime}")
