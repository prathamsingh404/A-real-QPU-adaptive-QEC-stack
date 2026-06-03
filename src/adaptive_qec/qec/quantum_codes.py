"""
Quantum code generators and hardware dynamic circuits.

Provides:
- [[4, 2, 2]] error-detecting code (protects against both X and Z Pauli errors).
- OpenQASM 3 Dynamic Circuits with sub-microsecond on-chip feedforward (`if_test`).
- Parameterized multi-round stabilizer syndrome extraction circuits.
- Exponential logical decay fitting utilities to extract logical error per round (epsilon_L).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def build_422_code_circuit(insert_dd: bool = True) -> QuantumCircuit:
    """
    Builds the [[4, 2, 2]] quantum code circuit.
    Encodes 2 logical qubits into 4 physical data qubits.
    Detects ANY single-qubit Pauli error (X, Y, or Z).

    Stabilizers:
        S_Z = Z0 Z1 Z2 Z3 (checks for bit-flip X errors)
        S_X = X0 X1 X2 X3 (checks for phase-flip Z errors)

    Args:
        insert_dd: If True, interleaves XY4 decoupling pulses during idling periods.

    Returns:
        QuantumCircuit ready for compilation to QPU ISA.
    """
    qr_data = QuantumRegister(4, name="data")
    qr_anc = QuantumRegister(2, name="anc")
    cr_syn = ClassicalRegister(2, name="syn")
    cr_data = ClassicalRegister(4, name="meas_data")

    qc = QuantumCircuit(qr_data, qr_anc, cr_syn, cr_data)

    # 1. State preparation: Encode |00>_L = (|0000> + |1111>) / sqrt(2)
    qc.reset(qr_data)
    qc.h(qr_data[0])
    qc.cx(qr_data[0], qr_data[1])
    qc.cx(qr_data[0], qr_data[2])
    qc.cx(qr_data[0], qr_data[3])

    if insert_dd:
        qc.barrier()
        for d in range(4):
            qc.x(qr_data[d])
            qc.y(qr_data[d])
            qc.x(qr_data[d])
            qc.y(qr_data[d])
        qc.barrier()

    # 2. Extract Z-stabilizer: S_Z = Z0 Z1 Z2 Z3 into Ancilla 0
    qc.reset(qr_anc[0])
    qc.cx(qr_data[0], qr_anc[0])
    qc.cx(qr_data[1], qr_anc[0])
    qc.cx(qr_data[2], qr_anc[0])
    qc.cx(qr_data[3], qr_anc[0])
    qc.measure(qr_anc[0], cr_syn[0])

    # 3. Extract X-stabilizer: S_X = X0 X1 X2 X3 into Ancilla 1
    qc.reset(qr_anc[1])
    qc.h(qr_anc[1])
    qc.cx(qr_anc[1], qr_data[0])
    qc.cx(qr_anc[1], qr_data[1])
    qc.cx(qr_anc[1], qr_data[2])
    qc.cx(qr_anc[1], qr_data[3])
    qc.h(qr_anc[1])
    qc.measure(qr_anc[1], cr_syn[1])

    # 4. Transversal Readout
    qc.barrier()
    qc.measure(qr_data, cr_data)
    return qc


def build_dynamic_feedforward_circuit(insert_error_probability: float = 0.0) -> QuantumCircuit:
    """
    Builds a dynamic circuit using Qiskit OpenQASM 3 `if_test`.
    Executes real-time sub-microsecond feedforward on IBM QPU controller electronics!

    Logic:
        1. Entangle data qubits into a Bell state (|00> + |11>)/sqrt(2).
        2. Measure parity check Z0 Z1 into ancilla.
        3. Real-time on-chip feedforward: IF syndrome == 1 THEN apply X correction immediately.
        4. Measure final data fidelity.
    """
    qr_data = QuantumRegister(2, name="data")
    qr_anc = QuantumRegister(1, name="anc")
    cr_syn = ClassicalRegister(1, name="syn")
    cr_data = ClassicalRegister(2, name="meas_data")

    qc = QuantumCircuit(qr_data, qr_anc, cr_syn, cr_data)

    # Prepare Bell state |Phi+> on data qubits
    qc.reset(qr_data)
    qc.h(qr_data[0])
    qc.cx(qr_data[0], qr_data[1])

    # Measure parity check Z0 Z1 into ancilla
    qc.reset(qr_anc[0])
    qc.cx(qr_data[0], qr_anc[0])
    qc.cx(qr_data[1], qr_anc[0])
    qc.measure(qr_anc[0], cr_syn[0])

    # ON-CHIP REAL-TIME ACTIVE FEEDFORWARD (Executed in ~300 ns on cryostat electronics)
    with qc.if_test((cr_syn[0], 1)):
        qc.x(qr_data[0])  # Active conditional Pauli correction

    qc.barrier()
    qc.measure(qr_data, cr_data)
    return qc


def build_multi_round_circuit(
    distance: int = 3,
    rounds: int = 2,
    insert_dd: bool = False,
) -> QuantumCircuit:
    """
    Build a multi-round stabilizer syndrome extraction circuit.
    Parameterized by distance d and round count R.
    """
    qr_data = QuantumRegister(distance, name="data")
    qr_anc = QuantumRegister(distance - 1, name="anc")
    cr_syn = [ClassicalRegister(distance - 1, name=f"syn{r}") for r in range(rounds)]
    cr_data = ClassicalRegister(distance, name="meas_data")

    qc = QuantumCircuit(qr_data, qr_anc, *cr_syn, cr_data)
    qc.reset(qr_data)

    for r in range(rounds):
        qc.barrier()
        qc.reset(qr_anc)
