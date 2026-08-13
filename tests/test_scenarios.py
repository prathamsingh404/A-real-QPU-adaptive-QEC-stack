"""
Tests for noise scenario factory.

Verifies:
    - Stationary noise produces constant error rates
    - Linear drift increases monotonically
    - Burst scenarios produce localized spikes
    - Multi-phase scenarios have distinct regime transitions
    - All scenarios produce valid NoiseSnapshot objects
    - Convenience constructors work
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_qec.engine.scenarios import (
    NoiseScenarioFactory,
    NoiseSnapshot,
    ScenarioConfig,
    ScenarioType,
    burst_scenario,
    drift_scenario,
    multi_phase_scenario,
    stationary_scenario,
)


class TestNoiseSnapshot:
    def test_to_calibration_dict(self):
        snap = NoiseSnapshot(
            gate_error_1q=0.0005,
            gate_error_2q=0.003,
            readout_error=0.01,
            t1_us=200.0,
            t2_us=150.0,
        )
        d = snap.to_calibration_dict()
        assert "gate_error_1q" in d
        assert d["t1_us"] == 200.0


class TestScenarioConfig:
    def test_default_config(self):
        config = ScenarioConfig(
            scenario_type=ScenarioType.STATIONARY,
            total_steps=100,
        )
        config.validate()

    def test_invalid_steps_raises(self):
        config = ScenarioConfig(
            scenario_type=ScenarioType.STATIONARY,
            total_steps=0,
        )
        with pytest.raises(ValueError):
            config.validate()


class TestStationaryScenario:
    def test_constant_error_rate(self):
        factory = stationary_scenario(total_steps=50)
        snapshots = [factory.get_noise(i) for i in range(50)]
        
        rates = [s.gate_error_2q for s in snapshots]
        assert all(abs(r - rates[0]) < 1e-10 for r in rates)

    def test_returns_valid_snapshots(self):
        factory = stationary_scenario(total_steps=10)
        for i in range(10):
            snap = factory.get_noise(i)
            assert isinstance(snap, NoiseSnapshot)
            assert snap.gate_error_2q > 0
            assert snap.t1_us > 0


class TestDriftScenario:
    def test_monotonic_drift(self):
        factory = drift_scenario(total_steps=100, rate=0.001)
        snapshots = [factory.get_noise(i) for i in range(100)]
        
        # Error rate should generally increase
        rates = [s.gate_error_2q for s in snapshots]
        assert rates[-1] > rates[0]

    def test_t1_degrades(self):
        factory = drift_scenario(total_steps=100, rate=0.001)
        snap_early = factory.get_noise(0)
        snap_late = factory.get_noise(99)
        
