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
        import time
        t_start = time.perf_counter()

        sampler = SamplerV2(backend=self._backend)
        job = sampler.run([transpiled], shots=shots)
        result = job.result()

        t_elapsed = time.perf_counter() - t_start

        # Extract measurement outcomes
        pub_result = result[0]
        # SamplerV2 returns BitArray; convert to numpy
        bitstrings = pub_result.data.meas
        outcomes = np.array(
            [[int(b) for b in bits] for bits in bitstrings.get_bitstrings()],
            dtype=np.uint8,
        )

        # Build counts dict
        counts: dict[str, int] = {}
        for bitstring in bitstrings.get_bitstrings():
            key = bitstring
            counts[key] = counts.get(key, 0) + 1

        logger.info(
            f"Experiment {experiment_id}: completed in {t_elapsed:.2f}s, "
            f"{len(counts)} unique outcomes"
        )

        return ExperimentResult(
            experiment_id=experiment_id,
            backend_name=self._backend_name,
            shots=shots,
            measurement_outcomes=outcomes,
            counts=counts,
            metadata={
                "transpiled_depth": transpiled.depth(),
                "transpiled_gate_count": transpiled.size(),
                "qubit_mapping": qubit_mapping,
            },
            execution_time_s=t_elapsed,
            job_id=job.job_id(),
        )

    def get_calibration(self) -> CalibrationSnapshot:
        """
        Capture current calibration snapshot from IBM backend.

        Extracts T1, T2, readout errors, gate errors, coupling map
        from backend.properties() and backend.target.
        """
        if self._backend is None:
            raise RuntimeError("Backend not connected. Call connect() first.")

        timestamp = datetime.now(timezone.utc).isoformat()
        properties = self._backend.properties()
        target = self._backend.target

        # Extract per-qubit calibration
        qubit_calibrations = []
        for qubit_idx in range(self._backend.num_qubits):
            qc = QubitCalibration(qubit_index=qubit_idx)

            if properties is not None:
                # Try to extract from properties
                try:
                    qubit_props = properties.qubit_property(qubit_idx)
                    if qubit_props:
                        t1_data = qubit_props.get("T1")
                        if t1_data:
                            qc.t1_us = t1_data[0] * 1e6 if t1_data[0] < 1 else t1_data[0]

                        t2_data = qubit_props.get("T2")
                        if t2_data:
                            qc.t2_us = t2_data[0] * 1e6 if t2_data[0] < 1 else t2_data[0]

                        readout_data = qubit_props.get("readout_error")
                        if readout_data:
                            qc.readout_error = readout_data[0]

                        freq_data = qubit_props.get("frequency")
                        if freq_data:
                            qc.frequency_ghz = freq_data[0] * 1e-9 if freq_data[0] > 1e6 else freq_data[0]
                except Exception as e:
                    logger.warning(f"Failed to extract properties for qubit {qubit_idx}: {e}")

            # Fallback to target if properties didn't work
            if target is not None and qc.t1_us is None:
                try:
                    qubit_props = target.qubit_properties
                    if qubit_props and qubit_idx < len(qubit_props):
                        props = qubit_props[qubit_idx]
                        if props:
                            if hasattr(props, 't1') and props.t1 is not None:
                                qc.t1_us = props.t1 * 1e6
                            if hasattr(props, 't2') and props.t2 is not None:
                                qc.t2_us = props.t2 * 1e6
                            if hasattr(props, 'frequency') and props.frequency is not None:
                                qc.frequency_ghz = props.frequency * 1e-9
                except Exception as e:
                    logger.warning(f"Failed to extract target props for qubit {qubit_idx}: {e}")

            qubit_calibrations.append(qc)

        # Extract gate calibrations
        gate_calibrations = []
        if target is not None:
            for gate_name in target.operation_names:
                for qargs in target.qargs_for_operation_name(gate_name):
                    try:
                        gate_props = target[gate_name][qargs]
                        if gate_props:
                            gc = GateCalibration(
                                gate_name=gate_name,
                                qubits=tuple(qargs),
                                error=gate_props.error,
                                gate_length_ns=(
                                    gate_props.duration * 1e9
                                    if gate_props.duration is not None
