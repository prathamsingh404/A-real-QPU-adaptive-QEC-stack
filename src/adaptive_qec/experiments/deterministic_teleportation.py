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

        # 1. Prepare target state on source qubit q[0]
        if self.target_state.theta != 0.0:
            qc.ry(self.target_state.theta, q[0])
        if self.target_state.phi != 0.0:
            qc.rz(self.target_state.phi, q[0])

        if self.regime == "swap_network":
            # Baseline: unitary swap network without mid-circuit measurement
            # SWAP q[0], q[1] then SWAP q[1], q[2]
            qc.swap(q[0], q[1])
            qc.swap(q[1], q[2])

        else:
            # 2. Distribute EPR pair between transit q[1] and destination q[2]
            qc.h(q[1])
            qc.cx(q[1], q[2])

            # 3. Bell-State Measurement (BSM) between source q[0] and transit q[1]
            qc.cx(q[0], q[1])
            qc.h(q[0])

            # Measure source and transit qubits into classical bits
            qc.measure(q[0], c_bsm[0])
            qc.measure(q[1], c_bsm[1])

            # 4. Optional Dynamical Decoupling on destination q[2] during BSM readout window
            if self.apply_dd:
                qc.x(q[2])
                qc.y(q[2])
                qc.x(q[2])
                qc.y(q[2])

            # 5. Dynamic Real-Time Feedforward Corrections on destination q[2]
            if self.regime == "dynamic":
                # If transit measured 1: apply Pauli-X
                with qc.if_test((c_bsm[1], 1)):
                    qc.x(q[2])
                # If source measured 1: apply Pauli-Z
                with qc.if_test((c_bsm[0], 1)):
                    qc.z(q[2])

        # 6. Quantum State Tomography rotation on destination qubit q[2]
        if basis == "X":
            qc.h(q[2])
        elif basis == "Y":
            qc.sdg(q[2])
            qc.h(q[2])

        # Final readout of destination qubit
        qc.measure(q[2], c_out[0])
        return qc

    def compute_fidelity_from_tomography(
        self,
        counts_x: Dict[str, int],
        counts_y: Dict[str, int],
        counts_z: Dict[str, int],
    ) -> Dict[str, Any]:
        """
        Reconstruct physical Bloch vector and calculate quantum state fidelity.

        Args:
            counts_x: Raw bitstring counts in X-basis.
            counts_y: Raw bitstring counts in Y-basis.
            counts_z: Raw bitstring counts in Z-basis.

        Returns:
            Dictionary containing expectation values, fidelity, classical bound test.
        """
        def _get_expectation(counts: Dict[str, int]) -> Tuple[float, int, int]:
            # Bitstring format in Qiskit: 'c_out c_bsm' e.g. '0 11' or '011'
            shots_0 = 0
            shots_1 = 0
            for bstr, freq in counts.items():
                cleaned = bstr.replace(" ", "")
                # c_out is the first (leftmost) bit in Qiskit reverse order
                # or last bit depending on formatting. We check destination bit.
                dest_bit = cleaned[0] if len(cleaned) == 3 else cleaned[-1]
                
                # If post-selected regime: filter out non-(0,0) BSM results
                if self.regime == "post_selected":
                    bsm_bits = cleaned[1:] if len(cleaned) == 3 else cleaned[:2]
                    if bsm_bits not in ("00", "0"):
                        continue

                if dest_bit == '0':
                    shots_0 += freq
                else:
                    shots_1 += freq

            valid_shots = shots_0 + shots_1
            if valid_shots == 0:
                return 0.0, 0, sum(counts.values())
            exp_val = (shots_0 - shots_1) / valid_shots
            return exp_val, valid_shots, sum(counts.values())

        exp_x, valid_x, total_x = _get_expectation(counts_x)
        exp_y, valid_y, total_y = _get_expectation(counts_y)
        exp_z, valid_z, total_z = _get_expectation(counts_z)

        measured_bloch = np.array([exp_x, exp_y, exp_z])
        ideal_bloch = np.array(self.target_state.ideal_bloch_vector)

        # Quantum state fidelity for single-qubit states:
        # F = (1 + r_ideal . r_measured) / 2
        fidelity = float(np.clip((1.0 + np.dot(ideal_bloch, measured_bloch)) / 2.0, 0.0, 1.0))
        total_valid = valid_x + valid_y + valid_z
        total_submitted = total_x + total_y + total_z
        yield_rate = total_valid / total_submitted if total_submitted > 0 else 0.0

        surpasses_classical = fidelity > CLASSICAL_TELEPORTATION_BOUND

        return {
            "target_state": self.target_state.name,
            "regime": self.regime,
            "measured_bloch_vector": [float(exp_x), float(exp_y), float(exp_z)],
            "ideal_bloch_vector": list(self.target_state.ideal_bloch_vector),
            "state_fidelity": fidelity,
            "classical_bound": CLASSICAL_TELEPORTATION_BOUND,
            "surpasses_classical_limit": surpasses_classical,
            "deterministic_yield_rate": yield_rate,
            "valid_shots": total_valid,
            "total_submitted_shots": total_submitted,
        }

    def compute_fidelity_from_bitstrings(
        self,
        out_x: List[str],
        bsm_x: Optional[List[str]],
        out_y: List[str],
        bsm_y: Optional[List[str]],
        out_z: List[str],
        bsm_z: Optional[List[str]],
    ) -> Dict[str, Any]:
        """
        Reconstruct fidelity directly from raw SamplerV2 bitstring arrays.
        """
        def _eval_basis(outs: List[str], bsms: Optional[List[str]]) -> Tuple[float, int, int]:
            s0 = 0
            s1 = 0
            total = len(outs)
            for i in range(total):
                if self.regime == "post_selected" and bsms is not None:
                    # In post-selected, keep only if bsm is '00'
                    if bsms[i].replace(" ", "") not in ("00", "0"):
                        continue
                b = outs[i].strip()
                if b == '0':
                    s0 += 1
                else:
                    s1 += 1
            valid = s0 + s1
            if valid == 0:
                return 0.0, 0, total
            return (s0 - s1) / valid, valid, total

        exp_x, val_x, tot_x = _eval_basis(out_x, bsm_x)
        exp_y, val_y, tot_y = _eval_basis(out_y, bsm_y)
        exp_z, val_z, tot_z = _eval_basis(out_z, bsm_z)

        measured_bloch = np.array([exp_x, exp_y, exp_z])
        ideal_bloch = np.array(self.target_state.ideal_bloch_vector)

        fidelity = float(np.clip((1.0 + np.dot(ideal_bloch, measured_bloch)) / 2.0, 0.0, 1.0))
        total_valid = val_x + val_y + val_z
        total_submitted = tot_x + tot_y + tot_z
        yield_rate = total_valid / total_submitted if total_submitted > 0 else 0.0

        return {
            "target_state": self.target_state.name,
            "regime": self.regime,
            "measured_bloch_vector": [float(exp_x), float(exp_y), float(exp_z)],
            "ideal_bloch_vector": list(self.target_state.ideal_bloch_vector),
            "state_fidelity": fidelity,
            "classical_bound": CLASSICAL_TELEPORTATION_BOUND,
            "surpasses_classical_limit": fidelity > CLASSICAL_TELEPORTATION_BOUND,
            "deterministic_yield_rate": yield_rate,
            "valid_shots": total_valid,
            "total_submitted_shots": total_submitted,
        }
