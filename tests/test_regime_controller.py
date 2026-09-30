import pytest
from adaptive_qec.controller.controller import (AdaptiveController, HardwareTelemetry, DecoderChoice, DDPattern)

def test_leakage_fast_path_at_d5():
    ctrl = AdaptiveController(leakage_threshold=0.15, distance_crossover=5)
    telem = HardwareTelemetry(leakage_fraction=0.18, code_distance=5)
    action = ctrl.select_action(telem)
    assert action.decoder == DecoderChoice.UNION_FIND
    assert action.dd_pattern == DDPattern.XY4
    assert 'leakage' in action.reason
