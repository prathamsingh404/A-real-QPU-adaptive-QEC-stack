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
        assert s0 is not None
        s0.t2_us = 20.0
        s1 = twin.get_qubit_state(1)
        assert s1 is not None
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

        # Qubit 0 should get protected because 30us idle on T2=20us creates severe dephasing
        assert schedule.qubit_sequences[0] == DDSequenceType.XY4
        assert 0 in schedule.protected_qubits

        # Qubit 1 should NOT get protected because 0.1us idle creates negligible dephasing
        assert schedule.qubit_sequences[1] == DDSequenceType.NONE
        assert 1 not in schedule.protected_qubits

        # Overall schedule checks
        assert schedule.total_pulses_inserted == 4  # 4 pulses for qubit 0 (XY4)
        assert schedule.estimated_noise_reduction > 0

    def test_cpmg_and_xy8_sequences(self):
        twin = HardwareDigitalTwin(num_qubits=2)
        s0 = twin.get_qubit_state(0)
        assert s0 is not None
        s0.t2_us = 30.0

        planner = AdaptiveDDPlanner(digital_twin=twin)

        # CPMG test
        sched_cpmg = planner.plan_schedule({0: 50.0}, preferred_sequence=DDSequenceType.CPMG)
        assert sched_cpmg.qubit_sequences[0] == DDSequenceType.CPMG
        assert sched_cpmg.total_pulses_inserted == 2

        # XY8 test
        sched_xy8 = planner.plan_schedule({0: 50.0}, preferred_sequence=DDSequenceType.XY8)
        assert sched_xy8.qubit_sequences[0] == DDSequenceType.XY8
        assert sched_xy8.total_pulses_inserted == 8

    def test_estimate_idle_map_and_apply(self):
        code = create_code("repetition", distance=3, rounds=3)
        circuit = code.generate_circuit()

        idle_map = AdaptiveDDPlanner.estimate_idle_map_from_circuit(circuit)
        assert len(idle_map) > 0

        planner = AdaptiveDDPlanner()
        schedule = planner.plan_schedule(idle_map)
        protected_circuit = planner.apply_dd_to_circuit(circuit, schedule)

        assert isinstance(protected_circuit, stim.Circuit)
        assert protected_circuit.num_detectors == circuit.num_detectors


class TestDigitalTwinDDCandidates:
    """Tests for Digital Twin get_dd_candidates method."""

    def test_get_dd_candidates(self):
        twin = HardwareDigitalTwin(num_qubits=3)
        q0 = twin.get_qubit_state(0)
        q1 = twin.get_qubit_state(1)
        q2 = twin.get_qubit_state(2)
        assert q0 is not None and q1 is not None and q2 is not None
        q0.t2_us = 10.0   # very bad T2
        q1.t2_us = 50.0   # moderate T2
        q2.t2_us = 500.0  # very good T2

        # During a 5us idle period:
        # q0 dephasing: 1 - exp(-5/10) = 0.393 > 0.001 -> Candidate
        # q1 dephasing: 1 - exp(-5/50) = 0.095 > 0.001 -> Candidate
        # q2 dephasing: 1 - exp(-5/500) = 0.010 > 0.001 -> Candidate
        candidates = twin.get_dd_candidates(idle_duration_us=5.0, pulse_error_budget=0.05)
        # With budget 0.05, only q0 and q1 exceed 0.05
        assert 0 in candidates
        assert 1 in candidates
        assert 2 not in candidates
