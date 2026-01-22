"""
Tests for the adaptive X/Z stabilizer scheduler.

Verifies:
    - Schedule types produce correct X:Z ratios
    - Scheduler responds to syndrome imbalance
    - Hysteresis prevents chattering at boundary
    - EWMA smoothing tracks defect rates correctly
    - Anti-chattering guard enforces minimum rounds
    - Schedule transitions are logged
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_qec.qec.schedules import (
    PREDEFINED_SCHEDULES,
    ScheduleType,
    StabilizerSchedule,
    create_custom_schedule,
    get_schedule,
    schedule_for_bias,
)
from adaptive_qec.qec.adaptive_scheduler import (
    AdaptiveSchedulerConfig,
    AdaptiveXZScheduler,
    DefectObservation,
)


# -----------------------------------------------------------------------
# StabilizerSchedule tests
# -----------------------------------------------------------------------

class TestStabilizerSchedule:
    def test_balanced_ratio(self):
        sched = get_schedule(ScheduleType.BALANCED)
        assert sched.x_fraction == 0.5
        assert sched.z_fraction == 0.5
        assert sched.ratio_xz == 1.0

    def test_x_heavy_ratio(self):
        sched = get_schedule(ScheduleType.X_HEAVY)
        assert sched.x_fraction == pytest.approx(2 / 3)
        assert sched.z_fraction == pytest.approx(1 / 3)

    def test_z_heavy_ratio(self):
        sched = get_schedule(ScheduleType.Z_HEAVY)
        assert sched.z_fraction == pytest.approx(2 / 3)
        assert sched.x_fraction == pytest.approx(1 / 3)

    def test_extreme_x_ratio(self):
        sched = get_schedule(ScheduleType.EXTREME_X)
        assert sched.x_fraction == pytest.approx(3 / 4)

    def test_extreme_z_ratio(self):
        sched = get_schedule(ScheduleType.EXTREME_Z)
        assert sched.z_fraction == pytest.approx(3 / 4)

    def test_round_type_cycling(self):
        sched = get_schedule(ScheduleType.X_HEAVY)  # XZX
        seq = sched.generate_sequence(9)
        assert seq == ["X", "Z", "X"] * 3

    def test_custom_schedule(self):
        sched = create_custom_schedule("XZZX")
        assert sched.schedule_type == ScheduleType.CUSTOM
        assert sched.x_fraction == 0.5
        assert sched.period == 4

    def test_invalid_pattern_raises(self):
        with pytest.raises(ValueError, match="invalid characters"):
            StabilizerSchedule(pattern="XYZ", schedule_type=ScheduleType.CUSTOM)

    def test_empty_pattern_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            StabilizerSchedule(pattern="", schedule_type=ScheduleType.CUSTOM)

    def test_to_dict(self):
        sched = get_schedule(ScheduleType.BALANCED)
        d = sched.to_dict()
        assert d["pattern"] == "XZ"
        assert d["period"] == 2


# -----------------------------------------------------------------------
# schedule_for_bias tests
# -----------------------------------------------------------------------

class TestScheduleForBias:
    def test_isotropic_returns_balanced(self):
        sched = schedule_for_bias(1.0)
        assert sched.schedule_type == ScheduleType.BALANCED

    def test_high_dephasing_returns_x_heavy(self):
        sched = schedule_for_bias(5.0)
        assert sched.schedule_type == ScheduleType.X_HEAVY

    def test_extreme_dephasing_returns_extreme_x(self):
        sched = schedule_for_bias(50.0)
        assert sched.schedule_type == ScheduleType.EXTREME_X

    def test_high_relaxation_returns_z_heavy(self):
        sched = schedule_for_bias(0.2)
        assert sched.schedule_type == ScheduleType.Z_HEAVY

    def test_extreme_relaxation_returns_extreme_z(self):
        sched = schedule_for_bias(0.05)
        assert sched.schedule_type == ScheduleType.EXTREME_Z

    def test_negative_bias_raises(self):
        with pytest.raises(ValueError):
            schedule_for_bias(-1.0)


# -----------------------------------------------------------------------
# AdaptiveSchedulerConfig tests
# -----------------------------------------------------------------------

class TestAdaptiveSchedulerConfig:
    def test_valid_config(self):
        config = AdaptiveSchedulerConfig()
        config.validate()  # Should not raise

    def test_invalid_ewma_alpha(self):
        config = AdaptiveSchedulerConfig(ewma_alpha=0.0)
        with pytest.raises(ValueError, match="ewma_alpha"):
            config.validate()

    def test_invalid_thresholds(self):
        config = AdaptiveSchedulerConfig(theta_enter=0.05, theta_exit=0.15)
        with pytest.raises(ValueError, match="theta_exit"):
            config.validate()


# -----------------------------------------------------------------------
# AdaptiveXZScheduler tests
# -----------------------------------------------------------------------

class TestAdaptiveXZScheduler:
    def test_initialization(self):
        scheduler = AdaptiveXZScheduler()
