from pathlib import Path
import json

import numpy as np
import pytest
import yaml

from adaptive_qec.controller.reweighting_gate import ConfidenceGatedReweightingController
from adaptive_qec.decoders.dem_calibration import calibrate_spitz_matching
from adaptive_qec.decoders.graph_fit import SpitzGraph
from adaptive_qec.experiments.causal_reweighting import circuit, data_qubits, matching, simulate_case

CFG = yaml.safe_load(Path("configs/suite.yaml").read_text())


@pytest.mark.parametrize("d", [3, 5, 7])
def test_injections_cover_repeated_ticks_and_target_data(d):
    base = circuit(d, "Z", "stationary", 0, CFG["base"])
    q = data_qubits(base)[0]
    hot = circuit(d, "Z", "hotspot", 0.04, CFG["base"], q)
    biased = circuit(d, "Z", "bias", 0.005, CFG["base"])
    ticks = sum(op.name == "TICK" for op in base)
    assert sum(op.name == "DEPOLARIZE1" and op.gate_args_copy() == [0.04] for op in hot) == ticks
    assert sum(op.name == "Z_ERROR" and len(op.targets_copy()) == d*d for op in biased) >= ticks
    assert biased.detector_error_model() != base.detector_error_model()
    assert hot.detector_error_model() != base.detector_error_model()


def test_packed_estimator_matches_original_including_boundary_and_observables():
    mem = circuit(3, "X", "stationary", 0, CFG["base"])
    base = matching(mem)
    x = mem.compile_detector_sampler(seed=4).sample(417)
    fit = SpitzGraph(base)
    new = fit.matching(fit.estimate(x), 0)
    old = calibrate_spitz_matching(base, x)
    for (u, v, a), (i, j, b) in zip(new.edges(), old.edges()):
        assert (u,v,a["fault_ids"]) == (i,j,b["fault_ids"])
        assert a["weight"] == pytest.approx(b["weight"])
    np.testing.assert_array_equal(new.decode_batch(x), old.decode_batch(x))


def test_reweighted_checkpoint_roundtrip():
    mem = circuit(3, "Z", "stationary", 0, CFG["base"])
    base = matching(mem)
    gate = ConfidenceGatedReweightingController(base, reference_defect_rate=0.00001,
                                               drift_z_threshold=0, dead_zone_epsilon=0, adaptation_tax=0)
    x = mem.compile_detector_sampler(seed=1).sample(1000)
    gate.evaluate_window(x)
    assert gate.is_reweighted
    restored = ConfidenceGatedReweightingController(base)
    restored.load_state(json.loads(json.dumps(gate.serialize_state())))
    assert restored.active_matching.edges() == gate.active_matching.edges()
    gate.reset_reference()
    assert gate.active_matching is base and gate.reference_shot_defect_rates is None


def test_information_timing_budget_and_reference_stream():
    rows = simulate_case((CFG, 3, "Z", 70000, "hotspot"))
    for r in rows:
        assert r["onset_block"] < r["recovery_block"]
        assert len(set(r["physical_rates"])) > 1
        for arm in ("raw","shrink","rolling","periodic","selective","legacy"):
            assert r["errors"][arm][0] == r["errors"]["static"][0]
            assert r["actions"][-1][arm] == "stay"
        for arm in ("periodic","selective"):
            assert sum(a[arm]=="fit" for a in r["actions"]) <= CFG["budget"]
    assert sum(rows[0]["errors"]["static"]) == sum(rows[1]["errors"]["static"])
