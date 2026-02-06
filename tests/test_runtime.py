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
        manager = ShotBudgetManager(
            BudgetConfig(max_total_shots=100000, max_session_shots=5000)
        )
        manager.record_usage(3000)
        manager.new_session()
        assert manager.session_used == 0
        assert manager.total_used == 3000  # Total carries over

    def test_utilization_tracking(self):
        manager = ShotBudgetManager(
            BudgetConfig(max_total_shots=10000, safety_margin=0)
        )
        manager.record_usage(5000)
        assert manager.utilization == pytest.approx(0.5)

    def test_summary(self):
        manager = ShotBudgetManager()
        summary = manager.summary()
        assert "total_used" in summary
        assert "utilization" in summary
        assert "estimated_cost" in summary

    def test_export_log(self):
        manager = ShotBudgetManager()
        manager.record_usage(1000, source="test")
        
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            path = f.name

        log_json = manager.export_log(path)
        data = json.loads(log_json)
        assert "allocations" in data
        assert len(data["allocations"]) == 1

    def test_safety_margin(self):
        manager = ShotBudgetManager(
            BudgetConfig(max_total_shots=10000, safety_margin=2000)
        )
        # Effective limit is 8000
        manager.record_usage(7000)
        assert manager.can_submit(500)
        assert not manager.can_submit(2000)


# -----------------------------------------------------------------------
# RuntimeLoopConfig tests
# -----------------------------------------------------------------------

class TestRuntimeLoopConfig:
    def test_valid_config(self):
        config = RuntimeLoopConfig(dry_run=True)
        config.validate()

    def test_invalid_shots(self):
        config = RuntimeLoopConfig(shots_per_batch=10)
        with pytest.raises(ValueError, match="shots_per_batch"):
            config.validate()


# -----------------------------------------------------------------------
# QiskitRuntimeLoop tests (dry run only)
# -----------------------------------------------------------------------

class TestQiskitRuntimeLoop:
    def test_dry_run_initialization(self):
        config = RuntimeLoopConfig(
            dry_run=True,
            shots_per_batch=500,
            max_batches=5,
            max_total_shots=10000,
        )
        loop = QiskitRuntimeLoop(config=config)
        assert loop.run_id is not None
        assert loop.total_shots == 0

    def test_dry_run_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RuntimeLoopConfig(
                dry_run=True,
                shots_per_batch=500,
                max_batches=3,
                max_total_shots=5000,
                output_dir=tmpdir,
            )
            loop = QiskitRuntimeLoop(config=config)
            loop.connect()
            results = loop.run()

            assert len(results) == 3
            assert loop.total_shots == 1500
            for r in results:
                assert isinstance(r, BatchResult)
                assert r.shots == 500

    def test_budget_integration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RuntimeLoopConfig(
                dry_run=True,
                shots_per_batch=1000,
                max_batches=10,
                max_total_shots=50000,
                output_dir=tmpdir,
            )
            budget = ShotBudgetManager(
                BudgetConfig(max_total_shots=3000, safety_margin=0)
            )
            loop = QiskitRuntimeLoop(
                config=config,
                budget_manager=budget,
            )
            loop.connect()
            results = loop.run()

            # Should stop at 3 batches (3000 shots)
            assert loop.total_shots <= 3000

    def test_convergence_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RuntimeLoopConfig(
                dry_run=True,
                shots_per_batch=1000,
                max_batches=100,
                max_total_shots=200000,
                convergence_threshold=0.5,  # Very loose threshold
                convergence_window=3,
                warmup_batches=2,
                output_dir=tmpdir,
            )
            loop = QiskitRuntimeLoop(config=config)
            loop.connect()
            results = loop.run()

            # Should stop before 100 batches due to convergence
            assert len(results) < 100

    def test_result_serialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RuntimeLoopConfig(
                dry_run=True,
                shots_per_batch=500,
                max_batches=2,
                max_total_shots=5000,
                output_dir=tmpdir,
            )
            loop = QiskitRuntimeLoop(config=config)
            loop.connect()
            loop.run()

            # Check that results file was created
            result_files = list(Path(tmpdir).glob("run_*.json"))
            assert len(result_files) == 1

            with open(result_files[0]) as f:
                data = json.load(f)
            assert data["summary"]["total_batches"] == 2
            assert data["summary"]["total_shots"] == 1000

    def test_summary(self):
        config = RuntimeLoopConfig(dry_run=True)
        loop = QiskitRuntimeLoop(config=config)
        summary = loop.summary()
        assert "run_id" in summary
        assert "total_shots" in summary
        assert "overall_ler" in summary
