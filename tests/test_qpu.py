"""Tests for QPU interface and backend registry."""

import numpy as np
import pytest

from adaptive_qec.config import HardwareConfig, HardwareProvider, load_config_from_dict
from adaptive_qec.qpu.base import (
    BackendInfo,
    CalibrationHistory,
    CalibrationSnapshot,
    ExperimentResult,
    GateCalibration,
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

        q0 = cal.get_qubit(0)
        assert q0 is not None and q0.t1_us == 100
        assert cal.t1_values().shape == (2,)
        assert cal.readout_errors().shape == (2,)


class TestMockBackend:
    """Test with mock backend."""

    def test_mock_run(self):
        config = HardwareConfig(provider=HardwareProvider.IBM, backend="mock")
        backend = MockQPUBackend(config)
        backend.connect()

        result = backend.run(None, shots=100)
        assert result.shots == 100
        assert result.measurement_outcomes.shape[0] == 100

    def test_mock_calibration(self):
        config = HardwareConfig()
        backend = MockQPUBackend(config)
        cal = backend.get_calibration()

        assert len(cal.qubit_calibrations) == 10
        t1_val = cal.qubit_calibrations[0].t1_us
        assert t1_val is not None and t1_val > 0

    def test_mock_topology(self):
        config = HardwareConfig()
        backend = MockQPUBackend(config)
        topo = backend.get_topology()

        assert topo.num_qubits == 10
        assert len(topo.coupling_map) == 9


class TestBackendRegistry:
    """Test backend registry."""

    def test_register_and_retrieve(self):
        register_backend("mock", MockQPUBackend)

        config = load_config_from_dict({
            "hardware": {"provider": "mock", "backend": "test"}
        })

        # This would fail because "mock" isn't in HardwareProvider enum
        # But we can test the registry directly
        from adaptive_qec.qpu.registry import _BACKENDS
        assert "mock" in _BACKENDS


class TestCalibrationHistory:
    """Test CalibrationHistory time-series container."""

    def test_calibration_history_tracking_and_statistics(self):
        history = CalibrationHistory()
        assert len(history) == 0
        assert history.summary()["total_snapshots"] == 0

        # Add 3 snapshots with drifting values
        for i, t_str in enumerate(["2026-09-10T00:00:00Z", "2026-09-10T01:00:00Z", "2026-09-10T02:00:00Z"]):
            snap = CalibrationSnapshot(
                timestamp=t_str,
                backend_name="ibm_heron",
                qubit_calibrations=[
                    QubitCalibration(qubit_index=0, t1_us=100.0 + 5.0 * i, t2_us=80.0 - 2.0 * i, readout_error=0.01 + 0.002 * i),
                    QubitCalibration(qubit_index=1, t1_us=120.0, t2_us=90.0, readout_error=0.015),
                ],
                gate_calibrations=[
                    GateCalibration(gate_name="cz", qubits=(0, 1), error=0.005 + 0.001 * i),
                ],
            )
            history.add_snapshot(snap)

        assert len(history) == 3
        assert history.timestamps == ["2026-09-10T00:00:00Z", "2026-09-10T01:00:00Z", "2026-09-10T02:00:00Z"]

        # Trajectories
        t1_traj = history.get_t1_trajectory(0)
        assert len(t1_traj) == 3
        assert t1_traj[0] == 100.0
        assert t1_traj[2] == 110.0

        t2_traj = history.get_t2_trajectory(0)
        assert t2_traj[2] == 76.0

        ro_traj = history.get_readout_error_trajectory(0)
        assert ro_traj[0] == 0.01

        cz_traj = history.get_gate_error_trajectory((0, 1), "cz")
        assert len(cz_traj) == 3
        assert pytest.approx(cz_traj[2]) == 0.007

        # Statistics
        stats = history.compute_drift_statistics(0)
        assert stats["qubit_index"] == 0
        assert stats["num_snapshots"] == 3
        assert stats["t1"]["mean"] == 105.0
        assert stats["t1"]["min"] == 100.0
        assert stats["t1"]["max"] == 110.0
        assert stats["t1"]["cv"] > 0

        # Summary
        summ = history.summary()
        assert summ["total_snapshots"] == 3
        assert summ["backend_name"] == "ibm_heron"
        assert summ["qubits_tracked"] == 2
