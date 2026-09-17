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
    QubitCalibration,
    TopologyInfo,
)

logger = logging.getLogger(__name__)


class IBMQuantumBackend(QPUBackend):
    """
    IBM Quantum backend via Qiskit Runtime.

    Uses SamplerV2 for circuit execution and captures full calibration
    data at experiment time.
    """

    def __init__(self, config: HardwareConfig) -> None:
        self._config = config
        self._service = None
        self._backend = None
        self._backend_name = config.backend

    def connect(self) -> None:
        """Connect to IBM Quantum via QiskitRuntimeService."""
        from qiskit_ibm_runtime import QiskitRuntimeService

        token = self._config.api_token
        if not token:
            raise RuntimeError(
                f"IBM Quantum API token not found. "
                f"Set the {self._config.api_token_env} environment variable "
                f"or add it to your .env file."
