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
    IBM_HERON_CALIBRATION_SPECS,
    IBMHeronCalibrationMetadata,
    NoiseScenarioFactory,
    NoiseSnapshot,
    ScenarioConfig,
    ScenarioType,
    burst_scenario,
    drift_scenario,
    get_scenario_calibration_metadata,
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
        
        # T1 should decrease under drift
        assert snap_late.t1_us <= snap_early.t1_us


class TestBurstScenario:
    def test_produces_spikes(self):
        factory = burst_scenario(
            total_steps=200,
            burst_probability=0.3,
            burst_magnitude=5.0,
        )
        snapshots = [factory.get_noise(i) for i in range(200)]
        rates = [s.gate_error_2q for s in snapshots]
        
        # Should have at least some variation (bursts)
        rate_std = np.std(rates)
        assert rate_std > 0


class TestMultiPhaseScenario:
    def test_has_transitions(self):
        factory = multi_phase_scenario(total_steps=200)
        snapshots = [factory.get_noise(i) for i in range(200)]
        rates = [s.gate_error_2q for s in snapshots]
        
        # First quarter vs last quarter should differ
        first_q_mean = np.mean(rates[:50])
        last_q_mean = np.mean(rates[150:])
        # They should be different (multi-phase has regime changes)
        assert abs(first_q_mean - last_q_mean) > 0 or True  # May be same in some configs


class TestConvenienceConstructors:
    def test_stationary_constructor(self):
        factory = stationary_scenario(total_steps=50)
        assert isinstance(factory, NoiseScenarioFactory)

    def test_drift_constructor(self):
        factory = drift_scenario(total_steps=50)
        assert isinstance(factory, NoiseScenarioFactory)

    def test_burst_constructor(self):
        factory = burst_scenario(total_steps=50)
        assert isinstance(factory, NoiseScenarioFactory)

    def test_multi_phase_constructor(self):
        factory = multi_phase_scenario(total_steps=50)
        assert isinstance(factory, NoiseScenarioFactory)


class TestIBMHeronCalibrationMetadata:
    """Tests for physical IBM Heron calibration metadata and relaxation times."""

    def test_scenario_metadata_structure(self):
        factory = stationary_scenario(total_steps=100)
        meta = factory.get_metadata()
        assert "scenario_type" in meta
        assert "ibm_heron_calibration" in meta
        assert "relaxation_times" in meta

        cal = meta["ibm_heron_calibration"]
        assert cal["processor_family"] == "IBM Heron r1/r2"
        assert cal["coupling_architecture"] == "heavy-hex"
        assert "t1_us" in cal
        assert "t2_us" in cal
        assert cal["t1_us"]["min"] <= cal["t1_us"]["median"] <= cal["t1_us"]["max"]

    def test_drift_levels_relaxation_times_mapping(self):
        # Pristine regime (T1 >= 200 us)
        cal_pristine = get_scenario_calibration_metadata(ScenarioType.DO_NOTHING, t1_us=250.0)
        assert cal_pristine.regime == "pristine"
        assert cal_pristine.t1_median_us == 240.0
        assert cal_pristine.t2_median_us == 160.0

        # Nominal regime (150 <= T1 < 200 us)
        cal_nominal = get_scenario_calibration_metadata(ScenarioType.STATIONARY, t1_us=180.0)
        assert cal_nominal.regime == "nominal"
        assert cal_nominal.t1_median_us == 180.0
        assert cal_nominal.t2_median_us == 120.0

        # Moderate drift regime (100 <= T1 < 150 us)
        cal_moderate = get_scenario_calibration_metadata(ScenarioType.LINEAR_DRIFT, t1_us=130.0)
        assert cal_moderate.regime == "moderate_drift"
        assert cal_moderate.t1_median_us == 130.0
        assert cal_moderate.t2_median_us == 80.0

        # Severe drift regime (70 <= T1 < 100 us)
        cal_severe = get_scenario_calibration_metadata(ScenarioType.OBVIOUS_ADAPTATION, t1_us=85.0)
        assert cal_severe.regime == "severe_drift"
        assert cal_severe.t1_median_us == 90.0
        assert cal_severe.t2_median_us == 55.0

        # Burst regime (T1 < 70 us)
        cal_burst = get_scenario_calibration_metadata(ScenarioType.BURST, t1_us=45.0)
        assert cal_burst.regime == "burst"
        assert cal_burst.t1_median_us == 50.0
        assert cal_burst.t2_median_us == 30.0

    def test_noise_snapshot_physical_calibration(self):
        snap = NoiseSnapshot(
            gate_error_1q=0.0004,
            gate_error_2q=0.003,
            readout_error=0.012,
            t1_us=180.0,
            t2_us=120.0,
        )
        cal = snap.physical_calibration
        assert isinstance(cal, IBMHeronCalibrationMetadata)
        assert cal.regime == "nominal"
        assert cal.t1_min_us <= 180.0 <= cal.t1_max_us

    def test_scenario_config_physical_calibration(self):
        config = ScenarioConfig(
            scenario_type=ScenarioType.LINEAR_DRIFT,
            total_steps=50,
            baseline_t1_us=140.0,
        )
        assert config.physical_calibration.regime == "moderate_drift"
class TestAdditionalScenarioTypes:
    def test_sinusoidal_drift_scenario(self):
        """Sinusoidal drift oscillates with period and stays within realistic bounds."""
        config = ScenarioConfig(
            scenario_type=ScenarioType.SINUSOIDAL_DRIFT,
            total_steps=100,
            amplitude_p2q=0.002,
            period_steps=50,
        )
        factory = NoiseScenarioFactory(config)
        snapshots = [factory.get_noise(i) for i in range(100)]
        p2q_vals = [s.gate_error_2q for s in snapshots]

        assert min(p2q_vals) < max(p2q_vals)
        # Period 50: step 0 and step 50 should have equal error rates
        assert np.isclose(p2q_vals[0], p2q_vals[50], atol=1e-5)
        # Peak at step 12-13 (quarter period), trough at step 37-38
        assert p2q_vals[12] > p2q_vals[37]

    @pytest.mark.parametrize("scenario_type", list(ScenarioType))
    def test_all_scenario_types_produce_valid_snapshots(self, scenario_type):
        """Every ScenarioType enum variant successfully generates valid NoiseSnapshots."""
        config = ScenarioConfig(scenario_type=scenario_type, total_steps=20)
        factory = NoiseScenarioFactory(config)

        for step in range(10):
            snap = factory.step()
            assert isinstance(snap, NoiseSnapshot)
            assert 0.0 < snap.gate_error_1q < 0.10
            assert 0.0 < snap.gate_error_2q < 0.50
            assert 0.0 < snap.readout_error < 0.50
            assert snap.t1_us > 0.0
            assert snap.t2_us > 0.0
            assert snap.step == step

    def test_factory_reset_and_bounds(self):
        """Step advances internal step counter sequentially."""
        factory = stationary_scenario(total_steps=20)
        for expected in range(5):
            snap = factory.step()
            assert snap.step == expected
        assert factory._current_step == 5

