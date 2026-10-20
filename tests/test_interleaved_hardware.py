"""
Tests for Real Hardware Artifact Integrity and Synthetic Demo Verification.
"""

from pathlib import Path
import json
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from synthetic_ab_demo import InterleavedHardwareRunner, InterleavedExperimentSummary


def test_synthetic_ab_demo_provenance():
    """Verify synthetic A/B demo executes and is explicitly flagged as synthetic."""
    runner = InterleavedHardwareRunner(
        backend_name="synthetic_heron_model",
        shots_per_block=100,
        num_blocks=4,
        seed=101,
    )
    summary = runner.run_interleaved_ab()

    assert summary.backend_name == "synthetic_heron_model"
    assert summary.total_blocks == 4
    assert len(summary.block_results) == 4
    assert len(summary.sha256_provenance) == 64


def test_real_hardware_artifacts_exist_and_valid():
    """Verify committed real IBM Quantum hardware artifacts exist with real job IDs."""
    qec_artifact = Path("data/hardware_results/ibm_marrakesh_qec_results.json")
    dynamic_artifact = Path("data/hardware_results/ibm_marrakesh_true_quantum_and_dynamic_results.json")

    assert qec_artifact.exists(), "Real IBM Marrakesh QEC artifact must exist"
    assert dynamic_artifact.exists(), "Real IBM Marrakesh dynamic circuit artifact must exist"

    with open(qec_artifact, "r", encoding="utf-8") as f:
        qec_data = json.load(f)
    assert qec_data["metadata"]["backend"] == "ibm_marrakesh"
    assert qec_data["metadata"]["job_id_unmitigated"].startswith("dashr9")

    with open(dynamic_artifact, "r", encoding="utf-8") as f:
        dyn_data = json.load(f)
    assert dyn_data["metadata"]["backend"] == "ibm_marrakesh"
    assert "true_quantum_422_code" in dyn_data["experiments"]
    assert dyn_data["experiments"]["true_quantum_422_code"]["job_id_unmitigated"].startswith("dasj9")
