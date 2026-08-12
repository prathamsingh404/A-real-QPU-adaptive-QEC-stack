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

    def get_calibration(self):
        qubits = [
            QubitCalibration(
                qubit_index=i,
                t1_us=100.0 + i,
                t2_us=80.0 + i,
                readout_error=0.01 + 0.001 * i,
            )
            for i in range(10)
        ]
        return CalibrationSnapshot(
            timestamp="2026-09-10T00:00:00Z",
            backend_name="mock_backend",
            qubit_calibrations=qubits,
            coupling_map=[(i, i + 1) for i in range(9)],
        )

    def get_topology(self):
        return TopologyInfo(
            num_qubits=10,
            coupling_map=[(i, i + 1) for i in range(9)],
        )

    def get_backend_info(self):
        return BackendInfo(
            name="mock_backend",
            provider="mock",
            num_qubits=10,
            topology_type="linear",
            version="1.0",
            status="online",
            max_shots=100000,
            max_circuits=300,
        )

    def is_available(self):
        return True


class TestQPUBase:
    """Test QPU base classes."""

    def test_topology_neighbors(self):
        topo = TopologyInfo(
            num_qubits=5,
            coupling_map=[(0, 1), (1, 2), (2, 3), (3, 4)],
        )
        assert set(topo.neighbors(1)) == {0, 2}
        assert set(topo.neighbors(0)) == {1}

    def test_calibration_snapshot(self):
        qubits = [
            QubitCalibration(qubit_index=0, t1_us=100, t2_us=80, readout_error=0.01),
            QubitCalibration(qubit_index=1, t1_us=120, t2_us=90, readout_error=0.02),
        ]
        cal = CalibrationSnapshot(
            timestamp="2026-01-01T00:00:00Z",
            backend_name="test",
            qubit_calibrations=qubits,
        )
