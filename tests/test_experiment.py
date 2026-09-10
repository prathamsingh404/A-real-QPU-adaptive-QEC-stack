"""Tests for the experiment data store and reproducibility."""

import json
from pathlib import Path

import numpy as np
import pytest

from adaptive_qec.data.models import (
    DecoderRecord,
    DetectorRecord,
    ExperimentMetrics,
    ExperimentRecord,
)
from adaptive_qec.data.store import ExperimentStore


class TestExperimentStore:
    """Test filesystem experiment store."""

    def _make_record(self, experiment_id="test_001"):
        return ExperimentRecord(
            experiment_id=experiment_id,
            timestamp="2026-09-10T00:00:00Z",
            backend="test_backend",
            provider="test",
            qubit_mapping=None,
            circuit_qasm="OPENQASM 3.0;",
            shots=1000,
            num_qubits_measured=10,
            measurement_outcomes=np.random.randint(0, 2, (1000, 10), dtype=np.uint8),
            counts={"0000000000": 500, "0000000001": 500},
            calibration_snapshot={"t1_mean_us": 100.0},
            software_versions={"python": "3.11"},
            compiler_config={"optimization_level": 1},
            config_snapshot={"qec": {"distance": 3}},
            git_commit="abc123",
        )

    def test_save_and_load(self, tmp_path):
        store = ExperimentStore(tmp_path)
        record = self._make_record()

        store.save_experiment(record)

        # Verify directory structure
        exp_dir = tmp_path / "test_001"
        assert exp_dir.exists()
        assert (exp_dir / "config.yaml").exists()
        assert (exp_dir / "circuit.qasm").exists()
        assert (exp_dir / "qpu_metadata.json").exists()
        assert (exp_dir / "calibration.json").exists()
        assert (exp_dir / "raw_results" / "measurement_outcomes.npy").exists()
        assert (exp_dir / "git_commit.txt").exists()

        # Load back
        loaded = store.load_experiment("test_001")
        assert loaded.experiment_id == "test_001"
        assert loaded.backend == "test_backend"
        assert loaded.measurement_outcomes.shape == (1000, 10)

    def test_save_detector_data(self, tmp_path):
        store = ExperimentStore(tmp_path)
        record = self._make_record()
        store.save_experiment(record)

        det_record = DetectorRecord(
            experiment_id="test_001",
            num_rounds=3,
            num_detectors=8,
            syndrome_tensor=np.random.randint(0, 2, (1000, 8), dtype=np.uint8),
            observable_flips=np.random.randint(0, 2, (1000, 1), dtype=np.uint8),
        )
        store.save_detector_data(det_record)

        loaded = store.load_detector_data("test_001")
        assert loaded.num_detectors == 8
        assert loaded.syndrome_tensor.shape == (1000, 8)

    def test_save_metrics(self, tmp_path):
        store = ExperimentStore(tmp_path)
        record = self._make_record()
        store.save_experiment(record)

        metrics = ExperimentMetrics(
            experiment_id="test_001",
            logical_error_rate=0.05,
            logical_error_rate_ci_low=0.04,
            logical_error_rate_ci_high=0.06,
            confidence_level=0.95,
        )
        store.save_metrics(metrics)

        loaded = store.load_metrics("test_001")
        assert loaded["logical_error_rate"] == 0.05

    def test_list_experiments(self, tmp_path):
        store = ExperimentStore(tmp_path)

        for i in range(3):
            record = self._make_record(f"exp_{i:03d}")
            store.save_experiment(record)

        experiments = store.list_experiments()
        assert len(experiments) == 3
        assert "exp_000" in experiments

    def test_delete_experiment(self, tmp_path):
        store = ExperimentStore(tmp_path)
        record = self._make_record()
        store.save_experiment(record)

        assert (tmp_path / "test_001").exists()
        store.delete_experiment("test_001")
        assert not (tmp_path / "test_001").exists()

    def test_load_nonexistent_raises(self, tmp_path):
        store = ExperimentStore(tmp_path)
        with pytest.raises(FileNotFoundError):
            store.load_experiment("nonexistent")
