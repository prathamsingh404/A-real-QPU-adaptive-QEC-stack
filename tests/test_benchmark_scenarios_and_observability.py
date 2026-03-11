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

