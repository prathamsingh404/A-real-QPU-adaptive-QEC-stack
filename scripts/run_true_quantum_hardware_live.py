"""
Live Execution of True Quantum Code [[4, 2, 2]] and Dynamic Feedforward on IBM Heron (ibm_marrakesh).

1. [[4, 2, 2]] True Quantum Error-Detecting Code:
   - Detects simultaneous X (bit-flip) and Z (phase-flip) errors.
   - Compares unmitigated baseline vs. XY4 dynamical decoupling on idle transmons.
2. Dynamic Feedforward Circuit:
   - Mid-circuit measurement with sub-microsecond on-chip classical branch (if_else).

All jobs submitted to ibm_marrakesh via Qiskit Runtime SamplerV2.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(".env")

import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from adaptive_qec.qec.quantum_codes import (
    build_422_code_circuit,
    build_dynamic_feedforward_circuit,
)
from adaptive_qec.experiments.adaptive_vs_static import two_proportion_z_test, wilson_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("hardware_live_runner")


def run_true_quantum_hardware_experiment():
    logger.info("Connecting to IBM Quantum Runtime Service...")
    service = QiskitRuntimeService(
        channel=os.getenv("IBM_QUANTUM_CHANNEL", "ibm_cloud"),
        token=os.getenv("IBM_QUANTUM_TOKEN"),
        instance=os.getenv("IBM_QUANTUM_INSTANCE"),
    )
    backend = service.backend("ibm_marrakesh")
    logger.info(f"Target QPU: {backend.name} (Heron r2 architecture, 156 qubits)")
    
    # 1. Build circuits
    qc_422_base = build_422_code_circuit(insert_dd=False)
    qc_422_dd = build_422_code_circuit(insert_dd=True)
    qc_dynamic = build_dynamic_feedforward_circuit()
    
    # 2. Transpile to native ISA (cz, sx, rz, reset, measure, if_else)
    logger.info("Transpiling circuits for heavy-hex ISA...")
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
    isa_422_base = pm.run(qc_422_base)
    isa_422_dd = pm.run(qc_422_dd)
    isa_dynamic = pm.run(qc_dynamic)
    
    logger.info(f"422 Base ISA ops:    {dict(isa_422_base.count_ops())}")
    logger.info(f"422 DD ISA ops:      {dict(isa_422_dd.count_ops())}")
    logger.info(f"Dynamic ISA ops:     {dict(isa_dynamic.count_ops())}")
    
    sampler = SamplerV2(mode=backend)
    shots = 1000
    
    # 3. Submit jobs
    logger.info(f"Submitting [[4, 2, 2]] Base circuit ({shots} shots)...")
    job_422_base = sampler.run([isa_422_base], shots=shots)
    job_id_422_base = job_422_base.job_id()
    logger.info(f"Submitted Job 1: {job_id_422_base}")
    
    logger.info(f"Submitting [[4, 2, 2]] DD circuit ({shots} shots)...")
    job_422_dd = sampler.run([isa_422_dd], shots=shots)
    job_id_422_dd = job_422_dd.job_id()
    logger.info(f"Submitted Job 2: {job_id_422_dd}")
    
    logger.info(f"Submitting Dynamic Feedforward circuit ({shots} shots)...")
    job_dynamic = sampler.run([isa_dynamic], shots=shots)
    job_id_dynamic = job_dynamic.job_id()
    logger.info(f"Submitted Job 3: {job_id_dynamic}")
    
    # 4. Wait for results
    logger.info("Waiting for execution on ibm_marrakesh...")
    res_422_base = job_422_base.result()
    logger.info(f"Job 1 ({job_id_422_base}) completed.")
    
    res_422_dd = job_422_dd.result()
    logger.info(f"Job 2 ({job_id_422_dd}) completed.")
    
    res_dynamic = job_dynamic.result()
    logger.info(f"Job 3 ({job_id_dynamic}) completed.")
    
    # 5. Parse [[4, 2, 2]] Results
    def parse_422_results(res):
        pub_res = res[0]
        syn_bits = pub_res.data.syn.get_bitstrings()
        data_bits = pub_res.data.meas_data.get_bitstrings()
        
        n_shots = len(data_bits)
        z_defects = 0  # syn[0] detects X bit-flips
        x_defects = 0  # syn[1] detects Z phase-flips
        total_errors_detected = 0
        code_space_violations = 0
        
        for i in range(n_shots):
            s = [int(b) for b in syn_bits[i]][::-1]
            d = [int(b) for b in data_bits[i]][::-1]
            
            z_det = s[0]
            x_det = s[1]
            if z_det == 1:
                z_defects += 1
            if x_det == 1:
                x_defects += 1
            if z_det == 1 or x_det == 1:
                total_errors_detected += 1
                
            # Parity of data qubits must be even for valid code space
            if sum(d) % 2 != 0:
                code_space_violations += 1
                
        return {
            "shots": n_shots,
            "z_stabilizer_defects": z_defects,
            "z_defect_rate": z_defects / n_shots,
            "x_stabilizer_defects": x_defects,
            "x_defect_rate": x_defects / n_shots,
            "total_detected_errors": total_errors_detected,
            "error_detection_rate": total_errors_detected / n_shots,
            "code_space_violations": code_space_violations,
            "code_space_fidelity": 1.0 - (code_space_violations / n_shots),
        }
        
    metrics_422_base = parse_422_results(res_422_base)
    metrics_422_dd = parse_422_results(res_422_dd)
    
    ci_base = wilson_ci(metrics_422_base["total_detected_errors"], shots)
    ci_dd = wilson_ci(metrics_422_dd["total_detected_errors"], shots)
    metrics_422_base["ci_95"] = list(ci_base)
    metrics_422_dd["ci_95"] = list(ci_dd)
    
    z_stat, p_val = two_proportion_z_test(
        metrics_422_base["total_detected_errors"], shots,
        metrics_422_dd["total_detected_errors"], shots,
    )
    
    # 6. Parse Dynamic Feedforward Results
    def parse_dynamic_results(res):
        pub_res = res[0]
        syn_bits = pub_res.data.syn.get_bitstrings()
        data_bits = pub_res.data.meas_data.get_bitstrings()
        
        n_shots = len(data_bits)
