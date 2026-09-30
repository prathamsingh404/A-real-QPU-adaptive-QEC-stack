import pytest
from adaptive_qec.controller.controller import (AdaptiveController, HardwareTelemetry, DecoderChoice, DDPattern)

def test_leakage_fast_path_at_d5():
    ctrl = AdaptiveController(leakage_threshold=0.15, distance_crossover=5)
    telem = HardwareTelemetry(leakage_fraction=0.18, code_distance=5)
    action = ctrl.select_action(telem)
    assert action.decoder == DecoderChoice.UNION_FIND
    assert action.dd_pattern == DDPattern.XY4
    assert 'leakage' in action.reason

def test_leakage_ignored_at_d3():
    ctrl = AdaptiveController(leakage_threshold=0.15, distance_crossover=5)
    telem = HardwareTelemetry(leakage_fraction=0.18, code_distance=3)
    action = ctrl.select_action(telem)
    assert action.decoder == DecoderChoice.MWPM

def test_burst_fast_path():
    ctrl = AdaptiveController()
    telem = HardwareTelemetry(burst_detected=True, code_distance=5)
    action = ctrl.select_action(telem)
    assert action.decoder == DecoderChoice.MWPM
    assert action.dd_pattern == DDPattern.XY4
    assert 'burst' in action.reason

def test_drift_mitigation():
    ctrl = AdaptiveController()
    telem = HardwareTelemetry(drift_magnitude=3.5, code_distance=5)
    action = ctrl.select_action(telem)
    assert action.dd_pattern == DDPattern.CPMG
    assert action.recalibrate_dem is True

def test_nominal_state():
    ctrl = AdaptiveController()
    telem = HardwareTelemetry(drift_magnitude=0.1, code_distance=5)
    action = ctrl.select_action(telem)
    assert action.decoder == DecoderChoice.MWPM
    assert action.dd_pattern == DDPattern.NONE
    assert action.recalibrate_dem is False

def test_fast_path_counter():
    ctrl = AdaptiveController(leakage_threshold=0.10, distance_crossover=3)
    telem = HardwareTelemetry(leakage_fraction=0.20, code_distance=3)
    ctrl.select_action(telem)
    assert ctrl.fast_path_count == 1

def test_controller_reset():
    ctrl = AdaptiveController(leakage_threshold=0.10, distance_crossover=3)
    telem = HardwareTelemetry(leakage_fraction=0.20, code_distance=3)
    ctrl.select_action(telem)
    ctrl.reset()
    assert ctrl.fast_path_count == 0
    assert ctrl.current_action.reason == 'reset'

def test_telemetry_defaults():
    telem = HardwareTelemetry()
    assert telem.drift_magnitude == 0.0
    assert telem.burst_detected is False
    assert telem.code_distance == 3
