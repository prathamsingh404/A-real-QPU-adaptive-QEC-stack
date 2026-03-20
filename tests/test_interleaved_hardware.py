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
