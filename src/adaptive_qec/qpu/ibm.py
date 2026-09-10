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
                f"Set the {self._config.api_token_env} environment variable."
            )

        logger.info("Connecting to IBM Quantum service...")
        self._service = QiskitRuntimeService(
            channel="ibm_quantum",
            token=token,
        )

        logger.info(f"Selecting backend: {self._backend_name}")
        self._backend = self._service.backend(self._backend_name)
        logger.info(
            f"Connected to {self._backend_name} — "
            f"{self._backend.num_qubits} qubits, "
            f"status: {self._backend.status().status_msg}"
        )

    def run(
        self,
        circuit: Any,
        shots: int,
        qubit_mapping: Optional[dict[int, int]] = None,
    ) -> ExperimentResult:
        """
        Execute a circuit on IBM QPU via SamplerV2.

        Args:
            circuit: A Qiskit QuantumCircuit.
            shots: Number of shots.
            qubit_mapping: Optional logical→physical qubit mapping.

        Returns:
            ExperimentResult with per-shot measurement outcomes.
        """
        from qiskit.compiler import transpile
        from qiskit_ibm_runtime import SamplerV2

        if self._backend is None:
            raise RuntimeError("Backend not connected. Call connect() first.")

        experiment_id = str(uuid.uuid4())[:12]
        logger.info(
            f"Experiment {experiment_id}: running {shots} shots "
            f"on {self._backend_name}"
        )

        # Transpile for the target backend
        initial_layout = None
        if qubit_mapping:
            initial_layout = [qubit_mapping.get(i, i) for i in range(circuit.num_qubits)]

        transpiled = transpile(
            circuit,
            backend=self._backend,
            initial_layout=initial_layout,
            optimization_level=1,
        )

        # Execute via SamplerV2
