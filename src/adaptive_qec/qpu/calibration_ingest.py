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
