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

    # Tier O0: Minimal
    env_o0 = SyntheticEnvironment(factory=factory, observability=ObservabilityLevel.O0_RAW_BITSTRINGS)
    obs_o0, _ = env_o0.step()
    assert "defect_rate" in obs_o0
    assert "p_2q" not in obs_o0
    assert "latent_true_p_2q" not in obs_o0

    # Tier O1: Macro Calibration
    env_o1 = SyntheticEnvironment(factory=factory, observability=ObservabilityLevel.O1_CALIBRATION_TELEMETRY)
    obs_o1, _ = env_o1.step()
    assert "p_2q" in obs_o1
    assert "t1_mean_us" in obs_o1
    assert "drift_magnitude" not in obs_o1
    assert "latent_true_p_2q" not in obs_o1

    # Tier O2: Deployable observables
    env_o2 = SyntheticEnvironment(factory=factory, observability=ObservabilityLevel.O2_DEPLOYABLE_OBSERVABLES)
    obs_o2, _ = env_o2.step()
    assert "drift_magnitude" in obs_o2
    assert "latent_true_p_2q" not in obs_o2

    # Tier O_oracle: Privileged Oracle
    env_oracle = SyntheticEnvironment(factory=factory, observability=ObservabilityLevel.O_ORACLE_FULL_LATENT)
    obs_oracle, _ = env_oracle.step()
    assert "latent_true_p_2q" in obs_oracle
    assert "latent_scenario" in obs_oracle
