"""
Pull and save live physical calibration snapshot from ibm_marrakesh.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

load_dotenv(".env")
from qiskit_ibm_runtime import QiskitRuntimeService

def pull_live_calibration() -> Path:
    service = QiskitRuntimeService(
        channel=os.getenv("IBM_QUANTUM_CHANNEL", "ibm_cloud"),
        token=os.getenv("IBM_QUANTUM_TOKEN"),
        instance=os.getenv("IBM_QUANTUM_INSTANCE"),
    )
    backend = service.backend("ibm_marrakesh")
    props = backend.properties()
    
    qubits_data = []
    t1_list = []
    t2_list = []
    ro_list = []
    for q in range(backend.num_qubits):
        try:
            t1 = props.t1(q)
            t1_us = t1 * 1e6 if t1 else None
        except Exception:
            t1_us = None

        try:
            t2 = props.t2(q)
            t2_us = t2 * 1e6 if t2 else None
        except Exception:
            t2_us = None

        try:
            ro = props.readout_error(q)
        except Exception:
            ro = None

        try:
            freq = props.frequency(q)
        except Exception:
            freq = None
        
        if t1_us is not None and not np.isnan(t1_us):
            t1_list.append(t1_us)
        if t2_us is not None and not np.isnan(t2_us):
            t2_list.append(t2_us)
        if ro is not None and not np.isnan(ro):
            ro_list.append(ro)
            
        qubits_data.append({
            "qubit": q,
            "t1_us": round(t1_us, 2) if t1_us else None,
            "t2_us": round(t2_us, 2) if t2_us else None,
            "frequency_ghz": round(freq / 1e9, 4) if freq else None,
            "readout_error": round(ro, 6) if ro else None,
        })
        
    gate_2q_errors = []
    if "cz" in backend.target:
        cz_map = backend.target["cz"]
        for pair, prop in cz_map.items():
            if prop and prop.error is not None and not np.isnan(prop.error):
                gate_2q_errors.append({
                    "gate": "cz",
                    "qubits": list(pair),
                    "error": float(prop.error),
                    "duration_ns": round(float(prop.duration) * 1e9, 1) if prop.duration else None,
                })

    gate_1q_errors = []
    if "sx" in backend.target:
        sx_map = backend.target["sx"]
        for q_tuple, prop in sx_map.items():
            if prop and prop.error is not None and not np.isnan(prop.error):
                gate_1q_errors.append(float(prop.error))
                
    two_q_err_vals = [g["error"] for g in gate_2q_errors if g["error"] > 0]
    one_q_err_vals = [e for e in gate_1q_errors if e > 0]
    
    now_str = datetime.now(timezone.utc).isoformat()
    calibration_doc = {
        "metadata": {
            "source": "IBM Quantum Platform (Live QPU Query)",
            "backend": backend.name,
            "processor_type": "Heron r2",
            "num_qubits": backend.num_qubits,
            "query_timestamp_utc": now_str,
            "backend_last_update_date": str(getattr(props, "last_update_date", "")),
            "status": "OPERATIONAL / MEASURED_LIVE",
        },
        "summary_statistics": {
            "mean_t1_us": round(float(np.mean(t1_list)), 2),
            "median_t1_us": round(float(np.median(t1_list)), 2),
            "std_t1_us": round(float(np.std(t1_list)), 2),
            "mean_t2_us": round(float(np.mean(t2_list)), 2),
            "median_t2_us": round(float(np.median(t2_list)), 2),
            "std_t2_us": round(float(np.std(t2_list)), 2),
            "mean_readout_error": round(float(np.mean(ro_list)), 6),
            "median_readout_error": round(float(np.median(ro_list)), 6),
            "mean_1q_gate_error": round(float(np.mean(one_q_err_vals)), 6) if one_q_err_vals else None,
            "median_1q_gate_error": round(float(np.median(one_q_err_vals)), 6) if one_q_err_vals else None,
            "mean_2q_gate_error": round(float(np.mean(two_q_err_vals)), 6) if two_q_err_vals else None,
            "median_2q_gate_error": round(float(np.median(two_q_err_vals)), 6) if two_q_err_vals else None,
            "num_2q_couplings_measured": len(gate_2q_errors),
        },
        "qubits": qubits_data,
        "gates_2q_sample": gate_2q_errors[:20],
    }
    
    out_dir = Path("data/calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ibm_marrakesh_live_snapshot.json"
    with open(out_file, "w") as f:
        json.dump(calibration_doc, f, indent=2)
        
    print(f"Saved live calibration snapshot to {out_file}")
    print("Summary:")
    for k, v in calibration_doc["summary_statistics"].items():
        print(f"  {k}: {v}")
    return out_file

if __name__ == "__main__":
    pull_live_calibration()
