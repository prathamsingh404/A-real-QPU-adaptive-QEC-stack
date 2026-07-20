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
            
            # 6 Detectors:
            # Round 0 ancillas: s0[0], s0[1]
            # Round 1 ancilla changes: s1[0] ^ s0[0], s1[1] ^ s0[1]
            # Final transversal data check: d[0]^d[1]^s1[0], d[1]^d[2]^s1[1]
            det0 = s0[0]
            det1 = s0[1]
            det2 = s1[0] ^ s0[0]
            det3 = s1[1] ^ s0[1]
            det4 = d[0] ^ d[1] ^ s1[0]
            det5 = d[1] ^ d[2] ^ s1[1]
            
            # Majority vote on data qubits for logical Z observable (0=clean, 1=flipped)
            logical_obs = 1 if sum(d) > 1 else 0
            
            syndromes.append([det0, det1, det2, det3, det4, det5])
            logical_observables.append(logical_obs)
            
        return np.array(syndromes, dtype=np.uint8), np.array(logical_observables, dtype=np.uint8)
        
    syn_base, obs_base = parse_job_results(res_base)
    syn_dd, obs_dd = parse_job_results(res_dd)
    
    # Build DEM using Stim for distance 3, 2 rounds repetition code
    stim_circuit = stim.Circuit.generated(
        "repetition_code:memory",
        distance=3,
        rounds=2,
        after_clifford_depolarization=0.015,
        before_round_data_depolarization=0.015,
        before_measure_flip_probability=0.015,
    )
    dem = stim_circuit.detector_error_model(decompose_errors=True)
    
    # Configure MWPM and Union-Find
    mwpm_decoder = pymatching.Matching.from_detector_error_model(dem)
    uf_decoder = UnionFindDecoder()
    uf_decoder.configure(dem=dem)
    
    # 1. MWPM Decoding
    mwpm_pred_base = mwpm_decoder.decode_batch(syn_base).flatten()
    mwpm_pred_dd = mwpm_decoder.decode_batch(syn_dd).flatten()
    
    mwpm_errors_base = int(np.sum(mwpm_pred_base != obs_base))
    mwpm_errors_dd = int(np.sum(mwpm_pred_dd != obs_dd))
    
    ler_base_mwpm = mwpm_errors_base / shots
    ler_dd_mwpm = mwpm_errors_dd / shots
    
    ci_base_mwpm = wilson_score_ci(mwpm_errors_base, shots)
    ci_dd_mwpm = wilson_score_ci(mwpm_errors_dd, shots)
    
    # 2. Union-Find Decoding
    uf_corr_base = uf_decoder.decode(syn_base)
    uf_corr_dd = uf_decoder.decode(syn_dd)
    
    uf_pred_base = uf_corr_base.observable_corrections.flatten()
    uf_pred_dd = uf_corr_dd.observable_corrections.flatten()
    
    uf_errors_base = int(np.sum(uf_pred_base != obs_base))
    uf_errors_dd = int(np.sum(uf_pred_dd != obs_dd))
    
    ler_base_uf = uf_errors_base / shots
    ler_dd_uf = uf_errors_dd / shots
    
    ci_base_uf = wilson_score_ci(uf_errors_base, shots)
    ci_dd_uf = wilson_score_ci(uf_errors_dd, shots)
    
    # Physical defect rates
    defect_rate_base = float(np.mean(syn_base))
    defect_rate_dd = float(np.mean(syn_dd))
    
    # Statistical significance: Base MWPM vs DD MWPM
    z_stat, p_val = two_proportion_z_test(mwpm_errors_dd, shots, mwpm_errors_base, shots)
    
    results = {
        "metadata": {
            "backend": "ibm_marrakesh",
            "processor": "Heron r2",
            "num_qubits": 156,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "code": "Repetition Code (Bit-flip Memory)",
            "distance": 3,
            "rounds": 2,
            "shots": shots,
            "job_id_unmitigated": job_id_base,
            "job_id_dd_mitigated": job_id_dd,
        },
        "hardware_metrics": {
            "defect_rate_unmitigated": round(defect_rate_base, 5),
            "defect_rate_dd": round(defect_rate_dd, 5),
            "mwpm": {
                "errors_unmitigated": mwpm_errors_base,
                "ler_unmitigated": round(ler_base_mwpm, 5),
                "ci_unmitigated": [round(ci_base_mwpm[0], 5), round(ci_base_mwpm[1], 5)],
                "errors_dd": mwpm_errors_dd,
                "ler_dd": round(ler_dd_mwpm, 5),
                "ci_dd": [round(ci_dd_mwpm[0], 5), round(ci_dd_mwpm[1], 5)],
            },
            "union_find": {
                "errors_unmitigated": uf_errors_base,
                "ler_unmitigated": round(ler_base_uf, 5),
                "ci_unmitigated": [round(ci_base_uf[0], 5), round(ci_base_uf[1], 5)],
                "errors_dd": uf_errors_dd,
                "ler_dd": round(ler_dd_uf, 5),
                "ci_dd": [round(ci_dd_uf[0], 5), round(ci_dd_uf[1], 5)],
            },
            "statistical_comparison": {
                "z_statistic": round(z_stat, 4),
                "p_value": round(p_val, 6),
                "significant_at_alpha_0_05": bool(p_val < 0.05),
            }
        }
    }
    
    out_dir = Path("data/hardware_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ibm_marrakesh_qec_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 75)
    print("LIVE HARDWARE EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 75)
    print(f"Backend:                         ibm_marrakesh (156 Transmons)")
    print(f"Job ID (Unmitigated Base):       {job_id_base}")
    print(f"Job ID (DD XY4 Mitigated):       {job_id_dd}")
    print(f"Total Shots per Arm:             {shots}")
    print(f"Physical Syndrome Defect Rate (Base): {defect_rate_base:.4f} ({defect_rate_base*100:.2f}%)")
    print(f"Physical Syndrome Defect Rate (DD):   {defect_rate_dd:.4f} ({defect_rate_dd*100:.2f}%)")
    print(f"MWPM LER (Unmitigated):          {ler_base_mwpm:.4f} ({mwpm_errors_base}/{shots}) CI {ci_base_mwpm}")
    print(f"MWPM LER (DD XY4):               {ler_dd_mwpm:.4f} ({mwpm_errors_dd}/{shots}) CI {ci_dd_mwpm}")
    print(f"Union-Find LER (Unmitigated):    {ler_base_uf:.4f} ({uf_errors_base}/{shots}) CI {ci_base_uf}")
    print(f"Union-Find LER (DD XY4):         {ler_dd_uf:.4f} ({uf_errors_dd}/{shots}) CI {ci_dd_uf}")
    print(f"Two-Proportion z-stat:           {z_stat:.4f}, p-value: {p_val:.6f}")
    print(f"Significant (alpha=0.05):        {p_val < 0.05}")
    print("=" * 75)
    
