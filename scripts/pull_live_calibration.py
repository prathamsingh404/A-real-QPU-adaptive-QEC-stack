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
