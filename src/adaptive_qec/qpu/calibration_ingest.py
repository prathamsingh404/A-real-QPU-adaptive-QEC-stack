"""
Calibration ingest pipeline for IBM Quantum backends.

Pulls live calibration data from the Qiskit Runtime service and
converts it into the internal CalibrationSnapshot format. This is
NOT simulation — it connects to real IBM backends and retrieves
actual hardware parameters.

Usage:
    ingest = CalibrationIngest(backend_name="ibm_marrakesh")
    snapshot = ingest.fetch()
    # snapshot contains real T1, T2, gate errors, coupling map

Environment:
    IBM_QUANTUM_TOKEN — IBM Quantum API token
    IBM_QUANTUM_INSTANCE — CRN or hub/group/project string
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from adaptive_qec.qpu.base import (
    BackendInfo,
    CalibrationSnapshot,
    GateCalibration,
    QubitCalibration,
    TopologyInfo,
)

logger = logging.getLogger(__name__)


class CalibrationIngest:
    """Pull live calibration from IBM Quantum via Qiskit Runtime.

    This uses the qiskit-ibm-runtime package to connect to real
    IBM backends and retrieve current calibration properties.

    Parameters
    ----------
    backend_name : str
        Name of the IBM backend (e.g., "ibm_marrakesh").
    channel : str
        IBM channel: "ibm_cloud" or "ibm_quantum".
    token_env : str
        Environment variable containing the API token.
    instance_env : str
        Environment variable containing the CRN/instance.
    """

    def __init__(
        self,
        backend_name: str = "ibm_marrakesh",
        channel: str = "ibm_cloud",
        token_env: str = "IBM_QUANTUM_TOKEN",
        instance_env: str = "IBM_QUANTUM_INSTANCE",
    ) -> None:
        self._backend_name = backend_name
        self._channel = channel
        self._token_env = token_env
        self._instance_env = instance_env
        self._service = None
        self._backend = None

    def _connect(self) -> None:
        """Establish connection to IBM Quantum Runtime service."""
        token = os.environ.get(self._token_env)
        instance = os.environ.get(self._instance_env)

        if not token:
            raise EnvironmentError(
                f"Missing API token.  Set {self._token_env} environment variable.\n"
                f"Get your token at https://quantum.ibm.com/"
            )

        try:
            from qiskit_ibm_runtime import QiskitRuntimeService

            self._service = QiskitRuntimeService(
                channel=self._channel,
                token=token,
                instance=instance,
            )
            self._backend = self._service.backend(self._backend_name)
            logger.info(f"Connected to IBM backend: {self._backend_name}")

        except ImportError:
            raise ImportError(
                "qiskit-ibm-runtime is required for live calibration.\n"
                "Install with: pip install qiskit-ibm-runtime"
            )
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to IBM backend {self._backend_name}: {e}"
            )

    def fetch(self) -> CalibrationSnapshot:
        """Fetch current calibration snapshot from hardware.

        Returns
        -------
        CalibrationSnapshot
            Contains real T1, T2, gate errors, readout errors,
            coupling map from the IBM backend.
        """
        if self._backend is None:
            self._connect()

        backend = self._backend
        properties = backend.properties()
        configuration = backend.configuration()

        timestamp = datetime.now(timezone.utc).isoformat()

        # Extract qubit calibrations
        qubit_cals: list[QubitCalibration] = []
        num_qubits = configuration.n_qubits if hasattr(configuration, 'n_qubits') else configuration.num_qubits

        for q_idx in range(num_qubits):
            t1 = self._get_qubit_property(properties, q_idx, "T1")
            t2 = self._get_qubit_property(properties, q_idx, "T2")
            ro_err = self._get_qubit_property(properties, q_idx, "readout_error")
            ro_len = self._get_qubit_property(properties, q_idx, "readout_length")
            freq = self._get_qubit_property(properties, q_idx, "frequency")
            anharm = self._get_qubit_property(properties, q_idx, "anharmonicity")

            # Convert T1/T2 from seconds to microseconds
            t1_us = t1 * 1e6 if t1 is not None else None
            t2_us = t2 * 1e6 if t2 is not None else None
            ro_len_ns = ro_len * 1e9 if ro_len is not None else None
            freq_ghz = freq * 1e-9 if freq is not None else None
            anharm_ghz = anharm * 1e-9 if anharm is not None else None

            qubit_cals.append(QubitCalibration(
                qubit_index=q_idx,
                t1_us=t1_us,
                t2_us=t2_us,
                readout_error=ro_err,
                readout_length_ns=ro_len_ns,
                frequency_ghz=freq_ghz,
                anharmonicity_ghz=anharm_ghz,
            ))

        # Extract gate calibrations
        gate_cals: list[GateCalibration] = []
        if hasattr(properties, 'gates') and properties.gates:
            for gate in properties.gates:
                gate_name = gate.gate if hasattr(gate, 'gate') else str(gate)
                qubits = tuple(gate.qubits) if hasattr(gate, 'qubits') else ()
                error = None
                length = None

                if hasattr(gate, 'parameters'):
                    for param in gate.parameters:
                        if hasattr(param, 'name'):
                            if param.name == 'gate_error':
                                error = param.value
                            elif param.name == 'gate_length':
                                length = param.value * 1e9 if param.value else None

                gate_cals.append(GateCalibration(
                    gate_name=gate_name,
                    qubits=qubits,
                    error=error,
                    gate_length_ns=length,
                ))

        # Extract coupling map
        coupling_map: list[tuple[int, int]] = []
        if hasattr(configuration, 'coupling_map') and configuration.coupling_map:
            coupling_map = [tuple(edge) for edge in configuration.coupling_map]

        snapshot = CalibrationSnapshot(
            timestamp=timestamp,
            backend_name=self._backend_name,
            qubit_calibrations=qubit_cals,
            gate_calibrations=gate_cals,
            coupling_map=coupling_map,
        )

        logger.info(
            f"Calibration fetched: {len(qubit_cals)} qubits, "
            f"{len(gate_cals)} gates, {len(coupling_map)} edges"
        )

        return snapshot

    def fetch_summary(self) -> dict[str, float]:
        """Fetch a summary dict suitable for the experiment harness.

        Returns mean T1, T2, gate errors, readout errors.
        """
        snapshot = self.fetch()

        t1_values = [q.t1_us for q in snapshot.qubit_calibrations if q.t1_us is not None]
        t2_values = [q.t2_us for q in snapshot.qubit_calibrations if q.t2_us is not None]
        ro_errors = [q.readout_error for q in snapshot.qubit_calibrations if q.readout_error is not None]
        sq_errors = [q.single_qubit_gate_error for q in snapshot.qubit_calibrations if q.single_qubit_gate_error is not None]

        # Two-qubit gate errors from gate calibrations
        tq_errors = [
            g.error for g in snapshot.gate_calibrations
            if g.error is not None and len(g.qubits) == 2
        ]

        return {
            "t1_mean_us": float(np.mean(t1_values)) if t1_values else 180.0,
            "t2_mean_us": float(np.mean(t2_values)) if t2_values else 120.0,
            "p_ro": float(np.mean(ro_errors)) if ro_errors else 0.012,
            "p_1q": float(np.mean(sq_errors)) if sq_errors else 0.0005,
            "p_2q": float(np.mean(tq_errors)) if tq_errors else 0.003,
            "t1_std_us": float(np.std(t1_values)) if len(t1_values) > 1 else 0.0,
            "t2_std_us": float(np.std(t2_values)) if len(t2_values) > 1 else 0.0,
            "p_ro_std": float(np.std(ro_errors)) if len(ro_errors) > 1 else 0.0,
            "p_2q_std": float(np.std(tq_errors)) if len(tq_errors) > 1 else 0.0,
            "num_qubits": len(snapshot.qubit_calibrations),
            "timestamp": snapshot.timestamp,
        }

    @staticmethod
    def _get_qubit_property(
        properties: Any,
        qubit_index: int,
        prop_name: str,
    ) -> Optional[float]:
        """Safely extract a qubit property from backend properties."""
        try:
            if hasattr(properties, 'qubit_property'):
                return properties.qubit_property(qubit_index, prop_name)
            elif hasattr(properties, 'qubits') and qubit_index < len(properties.qubits):
                qubit_props = properties.qubits[qubit_index]
                for prop in qubit_props:
                    if hasattr(prop, 'name') and prop.name == prop_name:
                        return prop.value
            return None
        except (IndexError, KeyError, AttributeError):
            return None


class OfflineCalibrationIngest:
    """Load calibration from a saved JSON snapshot (no network needed).

    For reproducibility and offline experiments.

    Parameters
    ----------
    snapshot_path : str
        Path to a JSON file containing a serialized CalibrationSnapshot.
    """

    def __init__(self, snapshot_path: str) -> None:
        import json
        with open(snapshot_path, "r") as f:
            data = json.load(f)
        self._data = data

    def fetch_summary(self) -> dict[str, float]:
        """Return calibration summary from saved data."""
        return {
            "t1_mean_us": self._data.get("t1_mean_us", 180.0),
            "t2_mean_us": self._data.get("t2_mean_us", 120.0),
            "p_ro": self._data.get("p_ro", 0.012),
            "p_1q": self._data.get("p_1q", 0.0005),
            "p_2q": self._data.get("p_2q", 0.003),
            "num_qubits": self._data.get("num_qubits", 156),
            "timestamp": self._data.get("timestamp", "offline"),
        }
