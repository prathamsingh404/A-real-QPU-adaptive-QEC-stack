"""Post hoc budget sensitivity: one fixed midpoint rebuild on the primary hotspot streams."""
import json
from pathlib import Path

import numpy as np
import yaml

from adaptive_qec.analysis.causal_results import contrast
from adaptive_qec.decoders.graph_fit import SpitzGraph
from adaptive_qec.experiments.causal_reweighting import circuit, data_qubits, matching

ROOT = Path(__file__).resolve().parents[1]


def compare():
    cfg = yaml.safe_load((ROOT / "configs/suite.yaml").read_text())
    primary = cfg["primary"]
    records = json.loads((ROOT / "data/suite_results.json").read_text())["trajectories"]
    cases = [r for r in records if all(r[k] == primary[k] for k in ("distance","basis","scenario","window"))]
    output = []
    for r in cases:
        d, basis, seed, kind, window = (r[k] for k in ("distance","basis","seed","scenario","window"))
        nominal = circuit(d, basis, "stationary", 0, cfg["base"])
        rng = np.random.default_rng(np.random.SeedSequence([seed,d,ord(basis),cfg["regimes"].index(kind)]))
        target = int(rng.choice(data_qubits(nominal)))
        start = int(rng.integers(12,21))
        stop = start + int(rng.integers(16,29))
        rate = float(rng.choice(cfg["amplitudes"][kind]))
        rng.integers(2**62)  # Independent reference seed consumed by the frozen simulation.
        if (start,stop,target) != (r["onset_block"],r["recovery_block"],r["target"]):
            raise ValueError("Stream reconstruction mismatch")
        x, y = [], []
        for block in range(cfg["blocks"]):
            mem = circuit(d,basis,kind,rate,cfg["base"],target) if start <= block < stop else nominal
            det, obs = mem.compile_detector_sampler(seed=int(rng.integers(2**62))).sample(
                cfg["block_shots"],separate_observables=True)
            x.append(det)
            y.append(obs)
        factory = matching(nominal)
        topology = SpitzGraph(factory)
        active = {name:factory for name in ("midpoint","midpoint_one_window","alarm_hold","alarm_one_window")}
        batches = np.concatenate(x).reshape(-1,window,nominal.num_detectors)
        truth = np.concatenate(y).reshape(len(batches),window,nominal.num_observables)
        errors = {name:[] for name in active}
        if sum(a["selective"] == "fit" for a in r["actions"]) != 1:
            raise ValueError("This post hoc comparison requires exactly one observed selective fit on each stream")
        alarm_fit = next(i for i,a in enumerate(r["actions"]) if a["selective"] == "fit")
        midpoint = len(batches)//2 - 1
        for i, batch in enumerate(batches):
            if np.count_nonzero(factory.decode_batch(batch) != truth[i]) != r["errors"]["static"][i]:
                raise ValueError("Factory error count differs on reconstructed stream")
            for name, graph in active.items():
                errors[name].append(int(np.count_nonzero(graph.decode_batch(batch) != truth[i])))
            if i in (midpoint, alarm_fit):
                fit = topology.matching(topology.estimate(batch), cfg["prior_fraction"])
                if i == midpoint:
                    active["midpoint"] = active["midpoint_one_window"] = fit
                if i == alarm_fit:
                    active["alarm_hold"] = active["alarm_one_window"] = fit
            if i == midpoint+1:
                active["midpoint_one_window"] = factory
            if i == alarm_fit+1:
                active["alarm_one_window"] = factory
        output.append(dict(seed=seed, midpoint_errors=errors["midpoint"], ablation_errors=errors,
                           selective_errors=r["errors"]["selective"],
                           selective_fits=sum(a["selective"]=="fit" for a in r["actions"]), midpoint_fits=1))
    deltas = [(sum(r["selective_errors"])-sum(r["midpoint_errors"]))/cases[0]["shots"] for r in output]
    return dict(stage="post hoc, primary streams reused", schedule="one fit after middle window", records=output,
                selective_minus_midpoint=contrast(deltas))


if __name__ == "__main__":
    result = compare()
    (ROOT/"data/midpoint_results.json").write_text(json.dumps(result,allow_nan=False)+"\n", encoding="utf8")
    print(sum(sum(r["midpoint_errors"]) for r in result["records"]), result["selective_minus_midpoint"])
