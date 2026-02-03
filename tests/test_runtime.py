"""
Tests for the Qiskit Runtime loop and shot budget manager.

Verifies:
    - Budget enforcement prevents over-allocation
    - Warning thresholds fire correctly
    - Session budget is independent of total budget
    - RuntimeLoop dry-run produces valid results
    - Convergence detection works
    - Result serialization round-trips
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from adaptive_qec.runtime.budget import (
    AllocationRecord,
    BudgetConfig,
    ShotBudgetManager,
)
from adaptive_qec.runtime.qiskit_loop import (
    BatchResult,
    QiskitRuntimeLoop,
    RuntimeLoopConfig,
)


# -----------------------------------------------------------------------
# ShotBudgetManager tests
# -----------------------------------------------------------------------

class TestBudgetConfig:
    def test_valid_config(self):
        config = BudgetConfig()
        config.validate()  # Should not raise

    def test_invalid_max_shots(self):
        config = BudgetConfig(max_total_shots=10)
        with pytest.raises(ValueError, match="max_total_shots"):
            config.validate()

    def test_invalid_threshold(self):
        config = BudgetConfig(warning_threshold=1.5)
        with pytest.raises(ValueError, match="warning_threshold"):
            config.validate()


class TestShotBudgetManager:
    def test_initialization(self):
        manager = ShotBudgetManager()
        assert manager.total_used == 0
        assert manager.total_remaining > 0

    def test_can_submit_within_budget(self):
        manager = ShotBudgetManager(BudgetConfig(max_total_shots=10000))
        assert manager.can_submit(1000)
        assert manager.can_submit(5000)

    def test_cannot_exceed_budget(self):
        manager = ShotBudgetManager(
            BudgetConfig(max_total_shots=10000, safety_margin=0)
        )
        assert not manager.can_submit(20000)

    def test_record_usage_updates_counters(self):
        manager = ShotBudgetManager(BudgetConfig(max_total_shots=10000))
        manager.record_usage(1000, source="test_batch_1")
        assert manager.total_used == 1000
        assert manager.session_used == 1000

    def test_budget_exhaustion(self):
        manager = ShotBudgetManager(
            BudgetConfig(max_total_shots=5000, safety_margin=0)
        )
        for i in range(4):
            assert manager.can_submit(1000)
            manager.record_usage(1000, source=f"batch_{i}")

        assert manager.total_used == 4000
        assert manager.can_submit(1000)
        manager.record_usage(1000)
        assert not manager.can_submit(1000)

    def test_session_reset(self):
