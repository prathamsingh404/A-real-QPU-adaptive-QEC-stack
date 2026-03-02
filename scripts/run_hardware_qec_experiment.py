"""
Comprehensive real-hardware QEC benchmark on ibm_marrakesh.
Executes stabilizer syndrome measurement circuits, compares unmitigated vs DD,
decodes with both MWPM and Union-Find, and generates analysis artifacts and figures.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv(".env")

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
import stim
import pymatching

from adaptive_qec.decoders.union_find import UnionFindDecoder
from adaptive_qec.analysis.significance import wilson_score_ci
from adaptive_qec.experiments.adaptive_vs_static import two_proportion_z_test


def build_qec_repetition_circuit(distance: int = 3, rounds: int = 2, insert_dd: bool = False) -> QuantumCircuit:
    """
    Build a distance-d repetition code with `rounds` syndrome stabilizer extractions.
    Uses d data qubits and d-1 ancilla qubits.
    """
    qr_data = QuantumRegister(distance, name="data")
    qr_anc = QuantumRegister(distance - 1, name="anc")
    cr_syn = [ClassicalRegister(distance - 1, name=f"syn{r}") for r in range(rounds)]
    cr_data = ClassicalRegister(distance, name="meas_data")
    
    qc = QuantumCircuit(qr_data, qr_anc, *cr_syn, cr_data)
    
    # Initialize logical |0> (all |0>)
    qc.reset(qr_data)
    
    for r in range(rounds):
        qc.barrier()
        qc.reset(qr_anc)
        
        # Entangle stabilizers Z_i Z_{i+1}
        for i in range(distance - 1):
            qc.cx(qr_data[i], qr_anc[i])
            qc.cx(qr_data[i + 1], qr_anc[i])
            
        if insert_dd:
            # Insert XY4 decoupling sequence on idle data qubits
            qc.barrier()
            for d in range(distance):
                qc.x(qr_data[d])
                qc.y(qr_data[d])
                qc.x(qr_data[d])
                qc.y(qr_data[d])
            qc.barrier()
            
        # Measure syndrome ancillas
        qc.measure(qr_anc, cr_syn[r])
        
    # Transversal readout of data qubits
    qc.barrier()
    qc.measure(qr_data, cr_data)
    return qc


def run_hardware_experiment():
    print("=" * 75)
    print("ADAPTIVE QEC - LIVE IBM MARRAKESH HARDWARE BENCHMARK")
    print("=" * 75)
    
    service = QiskitRuntimeService(
        channel=os.getenv("IBM_QUANTUM_CHANNEL", "ibm_cloud"),
        token=os.getenv("IBM_QUANTUM_TOKEN"),
        instance=os.getenv("IBM_QUANTUM_INSTANCE"),
    )
    backend = service.backend("ibm_marrakesh")
    print(f"Connected to backend: {backend.name} (156 transmons, Heron r2)")
    
    # Build unmitigated and DD circuits
    qc_base = build_qec_repetition_circuit(distance=3, rounds=2, insert_dd=False)
    qc_dd = build_qec_repetition_circuit(distance=3, rounds=2, insert_dd=True)
    
    print("\nTranspiling circuits for heavy-hex architecture...")
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
    isa_base = pm.run(qc_base)
    isa_dd = pm.run(qc_dd)
    
    print(f"Base circuit: depth={isa_base.depth()}, count_ops={dict(isa_base.count_ops())}")
    print(f"DD circuit:   depth={isa_dd.depth()}, count_ops={dict(isa_dd.count_ops())}")
    
    # Run jobs on real hardware
    shots = 500
    sampler = SamplerV2(mode=backend)
    print(f"\nSubmitting unmitigated QEC circuit ({shots} shots) to ibm_marrakesh...")
    job_base = sampler.run([isa_base], shots=shots)
    job_id_base = job_base.job_id()
    print(f"Base Job ID: {job_id_base}, status: {job_base.status()}")
    
    print(f"Submitting DD-mitigated QEC circuit ({shots} shots) to ibm_marrakesh...")
    job_dd = sampler.run([isa_dd], shots=shots)
    job_id_dd = job_dd.job_id()
    print(f"DD Job ID:   {job_id_dd}, status: {job_dd.status()}")
    
    # Wait for completion
    print("\nWaiting for hardware execution to complete on ibm_marrakesh...")
    res_base = job_base.result()
    print(f"Base job {job_id_base} DONE.")
    res_dd = job_dd.result()
    print(f"DD job {job_id_dd} DONE.")
    
    # Parse bitstrings
    def parse_job_results(job_res):
        pub_res = job_res[0]
        syn0_bits = pub_res.data.syn0.get_bitstrings()
        syn1_bits = pub_res.data.syn1.get_bitstrings()
        data_bits = pub_res.data.meas_data.get_bitstrings()
        
        num_shots = len(data_bits)
        syndromes = []
        logical_observables = []
        
        for i in range(num_shots):
            s0 = [int(b) for b in syn0_bits[i]][::-1]
            s1 = [int(b) for b in syn1_bits[i]][::-1]
            d = [int(b) for b in data_bits[i]][::-1]
