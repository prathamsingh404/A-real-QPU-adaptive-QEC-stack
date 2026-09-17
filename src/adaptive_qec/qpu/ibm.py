"""
IBM Quantum QPU adapter.

Implements the QPUBackend interface using Qiskit Runtime (qiskit-ibm-runtime).
Supports SamplerV2 for circuit execution and backend.properties() / backend.target
for calibration data.

This adapter handles:
- QiskitRuntimeService authentication
- Backend selection and connectivity
- Circuit transpilation with qubit mapping
- Calibration snapshot capture (T1, T2, readout, gate errors)
- Raw measurement result extraction
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from adaptive_qec.config import HardwareConfig
from adaptive_qec.qpu.base import (
    BackendInfo,
    CalibrationSnapshot,
    ExperimentResult,
    GateCalibration,
    QPUBackend,
