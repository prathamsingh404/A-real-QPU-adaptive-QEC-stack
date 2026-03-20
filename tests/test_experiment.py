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
