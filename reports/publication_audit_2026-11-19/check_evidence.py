"""Read-only checks for the November 2026 publication audit. Run from any directory."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pymatching
import yaml
from scipy import stats

ROOT = Path(os.environ.get("QEC_AUDIT_ROOT", Path(__file__).resolve().parents[2])).resolve()
if (ROOT / "archive/study_v2").exists():
    raise SystemExit("Historical audit: set QEC_AUDIT_ROOT to a checkout of 77861b1 (or c5a7a71 for the original run).")
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from adaptive_qec.controller.reweighting_gate import ConfidenceGatedReweightingController
from adaptive_qec.decoders.dem_calibration import calibrate_spitz_matching
from experiments.run_suite import build_circuit


def summarize_records(records):
    arms = ["static_mwpm", "cg_ccads", "naive_reweighted_mwpm", "oracle_mwpm"]
    groups = sorted({(r["distance"], r["scenario_id"], r["window_size"]) for r in records})
    print("distance scenario W | mean LER (static, gated, ungated, matched) | paired t p | decisions")
    for distance, scenario, window in groups:
        rows = [r for r in records if (r["distance"], r["scenario_id"], r["window_size"])
                == (distance, scenario, window)]
        rates = np.array([[r["arms"][arm]["errors"] / r["total_shots"] for arm in arms]
                          for r in rows])
        deltas = rates[:, 1] - rates[:, 0]
        actions = {k: sum(r["arms"]["cg_ccads"][k] for r in rows)
                   for k in ["switches", "stays", "rollbacks"]}
        p = stats.ttest_1samp(deltas, 0).pvalue if deltas.any() else float("nan")
        print(distance, scenario, window, rates.mean(axis=0).round(7), p, actions)


def check_hotspot_timing(config, distance):
    scenario = next(s for s in config["scenarios"] if s["id"] == "localized_hotspot")
    p_base = config["noise_regime"]["base_physical_error"]
    base = build_circuit(distance, distance, p_base, "stationary", p_base)
    hot = build_circuit(distance, distance, p_base, scenario["drift_type"],
                        scenario["drift_p"], scenario[f"hotspot_qubit_d{distance}"])
    anisotropy = build_circuit(distance, distance, p_base, "structural_anisotropy", 0.010)
    base_dem = base.detector_error_model(decompose_errors=True)
    print("anisotropy DEM equals stationary", distance,
          anisotropy.detector_error_model(decompose_errors=True) == base_dem)
    factory = pymatching.Matching.from_detector_error_model(base_dem)
    counts = []
    for seed in range(1000, 1030):
        samples, labels = hot.compile_detector_sampler(seed=seed).sample(
            shots=10000, separate_observables=True)
        gate = ConfidenceGatedReweightingController(factory)
        previous_gate = previous_ungated = factory
        errors = np.zeros(4, dtype=int)
        for start in range(0, 10000, 2000):
            x = samples[start:start + 2000]
            y = labels[start:start + 2000]
            _, current_gate, _ = gate.evaluate_window(x)
            for j, decoder in enumerate([factory, current_gate, previous_gate, previous_ungated]):
                errors[j] += np.count_nonzero(decoder.decode_batch(x) != y)
            previous_gate = current_gate
            previous_ungated = calibrate_spitz_matching(factory, x)
        counts.append(errors)
    errors = np.array(counts)
    print("hotspot d", distance, "order: static, same-window gate, next-window gate, next-window ungated")
    print("total errors / 300000", errors.sum(axis=0))
    for j in range(1, 4):
        delta = (errors[:, j] - errors[:, 0]) / 10000
        print("mean delta, unadjusted paired t p, counts by delta sign",
              delta.mean(), stats.ttest_1samp(delta, 0).pvalue, Counter(np.sign(delta)))


def check_shrinkage(config):
    """Exploratory baseline with equal weight on factory and previous-window edge probabilities."""
    p_base = config["noise_regime"]["base_physical_error"]
    base = build_circuit(3, 3, p_base, "stationary", p_base)
    factory = pymatching.Matching.from_detector_error_model(base.detector_error_model(decompose_errors=True))
    for scenario in config["scenarios"]:
        circuit = build_circuit(3, 3, p_base, scenario["drift_type"], scenario["drift_p"],
                                scenario.get("hotspot_qubit_d3"))
        for window in [200, 2000]:
            errors = 0
            for seed in range(1000, 1030):
                x, y = circuit.compile_detector_sampler(seed=seed).sample(10000, separate_observables=True)
                active = factory
                for start in range(0, len(x), window):
                    batch = x[start:start + window]
                    errors += np.count_nonzero(active.decode_batch(batch) != y[start:start + window])
                    if start + window == len(x):
                        break
                    estimate = calibrate_spitz_matching(factory, batch)
                    active = pymatching.Matching()
                    for u, v, prior in factory.edges():
                        empirical = (estimate.get_boundary_edge_data(u) if v is None
                                     else estimate.get_edge_data(u, v))
                        p = (prior["error_probability"] + empirical["error_probability"]) / 2
                        edge = dict(weight=np.log((1 - p) / p), error_probability=p,
                                    fault_ids=prior["fault_ids"])
                        if v is None:
                            active.add_boundary_edge(u, **edge)
                        else:
                            active.add_edge(u, v, **edge)
            print("half-prior shrinkage", scenario["id"], window, errors, errors / 300000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", type=Path, help="fresh suite JSON to compare by trajectory")
    parser.add_argument("--shrinkage", action="store_true", help="also run the exploratory 50:50 prior baseline")
    args = parser.parse_args()
    old = json.loads((ROOT / "data/suite_results.json").read_text())["trajectories"]
    summarize_records(old)
    if args.compare:
        new = json.loads(args.compare.read_text())["trajectories"]
        def keyed(records):
            return {(r["seed"], r["distance"], r["scenario_id"], r["window_size"]): r
                    for r in records}
        a, b = keyed(old), keyed(new)
        print("same trajectory keys", a.keys() == b.keys())
        print("identical trajectory records", sum(r == b.get(k) for k, r in a.items()), "/", len(a))
        print("fresh suite")
        summarize_records(new)
    config = yaml.safe_load((ROOT / "configs/suite.yaml").read_text())
    for distance in [3, 5]:
        check_hotspot_timing(config, distance)
    if args.shrinkage:
        check_shrinkage(config)
    lock = json.loads((ROOT / "paper.lock").read_text())["files"]
    for path, expected in lock.items():
        blob = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)
        print("lock matches raw Git blob", path, hashlib.sha256(blob).hexdigest() == expected)


if __name__ == "__main__":
    main()
