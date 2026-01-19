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

