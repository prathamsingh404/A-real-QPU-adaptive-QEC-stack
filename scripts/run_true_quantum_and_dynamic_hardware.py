"""
Advanced Quantum Hardware Experiments on ibm_marrakesh:
1. True Quantum [[4, 2, 2]] Error-Detecting Code (protects against BOTH X and Z errors).
2. On-Chip Real-Time Dynamic Circuit Feedforward (sub-microsecond if_test feedback).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


def build_422_code_circuit(insert_dd: bool = True) -> QuantumCircuit:
    """
    Builds the [[4, 2, 2]] quantum code circuit.
    Encodes 2 logical qubits into 4 physical qubits.
    Detects ANY single-qubit Pauli error (X, Y, or Z).
    
    Stabilizers:
        S_Z = Z0 Z1 Z2 Z3 (checks for bit-flips X)
        S_X = X0 X1 X2 X3 (checks for phase-flips Z)
    """
    qr_data = QuantumRegister(4, name="data")
    qr_anc = QuantumRegister(2, name="anc")
    cr_syn = ClassicalRegister(2, name="syn")
    cr_data = ClassicalRegister(4, name="meas_data")
    
    qc = QuantumCircuit(qr_data, qr_anc, cr_syn, cr_data)
    
    # 1. State preparation: Encode |00>_L = (|0000> + |1111>) / sqrt(2)
    qc.h(qr_data[0])
    qc.cx(qr_data[0], qr_data[1])
    qc.cx(qr_data[0], qr_data[2])
    qc.cx(qr_data[0], qr_data[3])
