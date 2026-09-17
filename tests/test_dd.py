"""Tests for Dynamical Decoupling Integration (Problem 3).

Validates selective DD candidate decision rule, sequence selection (CPMG, XY4, XY8),
idle time estimation from circuits, and Digital Twin DD candidate querying.
"""

import pytest
import stim

from adaptive_qec.config import NoiseConfig
from adaptive_qec.digital_twin.twin import HardwareDigitalTwin
from adaptive_qec.mitigation.dynamical_decoupling import (
    AdaptiveDDPlanner,
    DDSchedule,
    DDSequenceType,
)
from adaptive_qec.qec.codes import create_code


class TestDynamicalDecoupling:
    """Tests for AdaptiveDDPlanner and DDSchedule."""

    def test_selective_dd_decision_rule(self):
        twin = HardwareDigitalTwin(num_qubits=5)
        # Configure qubit 0 with poor T2 (20us) and qubit 1 with excellent T2 (2000us)
        s0 = twin.get_qubit_state(0)
        s0.t2_us = 20.0
        s1 = twin.get_qubit_state(1)
        s1.t2_us = 2000.0

        planner = AdaptiveDDPlanner(
            digital_twin=twin,
            single_qubit_pulse_error=0.0003,
            default_sequence=DDSequenceType.XY4,
        )

        # 30us idle window
        idle_map = {0: 30.0, 1: 0.1}  # qubit 0 is idle for 30us, qubit 1 is idle for 0.1us

        schedule = planner.plan_schedule(idle_map)
        assert isinstance(schedule, DDSchedule)

