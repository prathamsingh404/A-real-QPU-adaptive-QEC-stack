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
