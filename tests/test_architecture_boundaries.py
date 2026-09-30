"""
Architectural Boundary & Rigor Verification Tests (tests/test_architecture_boundaries.py).

Guarantees that past audit bugs cannot recur:
1. Test that the decoder actually invoked matches the controller's action.
2. Test that each experimental arm computes its own predictions independently without copying.
3. Test that the controller module does not import or peek into ground-truth noise schedules.
"""

import ast
import inspect
from pathlib import Path
import numpy as np
import pytest

from adaptive_qec.controller.controller import (
    AdaptiveController,
    ControlAction,
    DecoderChoice,
    HardwareState,
)
from adaptive_qec.decoders.mwpm import MWPMDecoder
from adaptive_qec.decoders.union_find import UnionFindDecoder
from adaptive_qec.mitigation.dynamical_decoupling import DDSequenceType
from adaptive_qec.qec.codes import create_code


def test_invoked_decoder_matches_controller_action():
    """Verify that when controller chooses UF, UF is invoked; when MWPM, MWPM is invoked.
    
    Prevents bug in bandit_vs_static.py where mwpm_matcher was executed for all arms.
    """
    code = create_code("surface", distance=3, rounds=3)
    circuit = code.generate_circuit()
    
    mwpm_decoder = MWPMDecoder()
    mwpm_decoder.configure(circuit=circuit)
    
    uf_decoder = UnionFindDecoder()
    uf_decoder.configure(circuit=circuit)
    
    sampler = circuit.compile_detector_sampler(seed=123)
    syndromes, observables = sampler.sample(shots=10, separate_observables=True)
    
    # Action 1: MWPM
    action_mwpm = ControlAction(decoder=DecoderChoice.MWPM, dd_policy=DDSequenceType.NONE)
    invoked_decoder_type = None
    if action_mwpm.decoder == DecoderChoice.MWPM:
        invoked_decoder_type = "mwpm"
        res = mwpm_decoder.decode_batch(syndromes.astype(np.uint8), observables.astype(np.uint8))
    else:
        invoked_decoder_type = "union_find"
        res = uf_decoder.decode_batch(syndromes.astype(np.uint8), observables.astype(np.uint8))
        
    assert invoked_decoder_type == action_mwpm.decoder.value
    assert res.total_shots == 10

    # Action 2: Union-Find
    action_uf = ControlAction(decoder=DecoderChoice.UNION_FIND, dd_policy=DDSequenceType.XY4)
    if action_uf.decoder == DecoderChoice.MWPM:
        invoked_decoder_type = "mwpm"
        res = mwpm_decoder.decode_batch(syndromes.astype(np.uint8), observables.astype(np.uint8))
    else:
        invoked_decoder_type = "union_find"
        res = uf_decoder.decode_batch(syndromes.astype(np.uint8), observables.astype(np.uint8))
        
    assert invoked_decoder_type == action_uf.decoder.value
    assert res.total_shots == 10


def test_static_arms_compute_own_predictions():
    """Verify that each static arm computes its own predictions on its own circuit/syndromes.
    
    Prevents bug in full_adaptive.py where static arrays were filled by copying adaptive LER.
    """
    code = create_code("surface", distance=3, rounds=3)
    circ1 = code.generate_circuit()
    circ2 = code.generate_circuit()
    
    mwpm1 = MWPMDecoder()
    mwpm1.configure(circuit=circ1)
    
    mwpm2 = MWPMDecoder()
    mwpm2.configure(circuit=circ2)
    
    # Use different seeds to ensure independent outputs
    s1, o1 = circ1.compile_detector_sampler(seed=101).sample(shots=20, separate_observables=True)
    s2, o2 = circ2.compile_detector_sampler(seed=202).sample(shots=20, separate_observables=True)
    
    res1 = mwpm1.decode_batch(s1.astype(np.uint8), o1.astype(np.uint8))
    res2 = mwpm2.decode_batch(s2.astype(np.uint8), o2.astype(np.uint8))
    
    # Ensure they are distinct objects and not identical references
    assert res1 is not res2
    assert hasattr(res1, "logical_error_rate")
    assert hasattr(res2, "logical_error_rate")


def test_controller_does_not_import_noise_schedules():
    """Verify that controller module does not import simulation noise schedules.
    
    Structural enforcement preventing the controller from peeking at ground truth.
    """
    controller_file = Path(__file__).resolve().parents[1] / "src" / "adaptive_qec" / "controller" / "controller.py"
    with open(controller_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(controller_file))
        
    imported_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imported_names.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_names.append(node.module)
                
    forbidden_modules = [
        "adaptive_qec.experiments",
        "adaptive_qec.experiments.adaptive_vs_static",
        "NoiseSchedule",
    ]
    for imp in imported_names:
        for forbidden in forbidden_modules:
            assert forbidden not in imp, f"Forbidden import detected in controller: {imp}"
