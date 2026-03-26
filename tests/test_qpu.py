"""Tests for QPU interface and backend registry."""

import numpy as np
import pytest

from adaptive_qec.config import HardwareConfig, HardwareProvider, load_config_from_dict
from adaptive_qec.qpu.base import (
    BackendInfo,
    CalibrationSnapshot,
    ExperimentResult,
    QPUBackend,
    QubitCalibration,
    TopologyInfo,
)
from adaptive_qec.qpu.registry import get_backend, register_backend


class MockQPUBackend(QPUBackend):
    """Mock QPU backend for testing without real hardware."""

    def __init__(self, config):
        self._config = config

    def connect(self):
        pass

    def run(self, circuit, shots, qubit_mapping=None):
        # Generate random measurement outcomes
        num_qubits = 10
        outcomes = np.random.randint(0, 2, size=(shots, num_qubits), dtype=np.uint8)
        counts = {}
        for row in outcomes:
            key = "".join(str(b) for b in row)
            counts[key] = counts.get(key, 0) + 1

        return ExperimentResult(
            experiment_id="mock_001",
            backend_name="mock_backend",
            shots=shots,
            measurement_outcomes=outcomes,
            counts=counts,
            execution_time_s=0.1,
        )
