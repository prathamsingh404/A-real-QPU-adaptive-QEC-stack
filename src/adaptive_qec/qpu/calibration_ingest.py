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
