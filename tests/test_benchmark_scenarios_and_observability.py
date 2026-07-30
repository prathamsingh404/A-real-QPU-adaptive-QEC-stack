"""
Tests for 12 Benchmark Scenarios and 4 Observability Tiers (tests/test_benchmark_scenarios_and_observability.py).

Validates TASK-063 through TASK-075:
- 12 predefined scenarios (stationary, drift, burst, do_nothing, obvious_adaptation,
  too_fast_to_adapt, switching_trap, hidden_transition, recovery, persistent_leakage,
  deceptive_noise, multi_phase)
- 4 observability levels (O0, O1, O2, O_oracle)
"""

import pytest

from adaptive_qec.engine.scenarios import (
    NoiseScenarioFactory,
    ObservabilityLevel,
    ScenarioConfig,
    ScenarioType,
    SyntheticEnvironment,
)


def test_all_twelve_scenarios_generate_valid_snapshots():
    """Verify that all 12 scenario types produce valid physical noise snapshots."""
    for stype in ScenarioType:
        config = ScenarioConfig(scenario_type=stype, total_steps=30)
        factory = NoiseScenarioFactory(config)
        snap0 = factory.get_noise(0)
        snap15 = factory.get_noise(15)
        snap25 = factory.get_noise(25)

        assert 0.0 < snap0.p_2q < 0.20
        assert 0.0 < snap15.p_2q < 0.20
        assert 0.0 < snap25.p_2q < 0.20
        assert snap0.t1_mean_us > 0.0
        assert snap0.t2_mean_us > 0.0


def test_synthetic_environment_observability_tiers():
    """Verify that information is gated strictly across O0, O1, O2, and O_oracle."""
    config = ScenarioConfig(scenario_type=ScenarioType.DO_NOTHING, total_steps=20)
    factory = NoiseScenarioFactory(config)
