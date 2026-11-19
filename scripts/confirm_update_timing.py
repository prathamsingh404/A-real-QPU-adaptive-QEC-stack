"""Independent test of one-lookahead-window update timing against a trained calendar schedule."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from adaptive_qec.analysis.causal_results import contrast
from adaptive_qec.decoders.graph_fit import SpitzGraph
from adaptive_qec.experiments.causal_reweighting import circuit, data_qubits, drift_alarm, matching

ROOT = Path(__file__).resolve().parents[1]


def run_case(d, seed, cfg):
    window, basis, kind = 2000, "Z", "hotspot"
    rng = np.random.default_rng(np.random.SeedSequence([seed,d,ord(basis),cfg["regimes"].index(kind)]))
    nominal = circuit(d,basis,"stationary",0,cfg["base"])
    factory = matching(nominal)
    prior = SpitzGraph(factory)
    target = int(rng.choice(data_qubits(nominal)))
    start = int(rng.integers(12,21))
    stop = start + int(rng.integers(16,29))
    level = float(rng.choice(cfg["amplitudes"][kind]))
    ref = nominal.compile_detector_sampler(seed=int(rng.integers(2**62))).sample(cfg["reference_shots"])
    reference = ref.sum(axis=0,dtype=np.int64)
    hot = circuit(d,basis,kind,level,cfg["base"],target)
    x, y = [], []
    for block in range(cfg["blocks"]):
        mem = hot if start <= block < stop else nominal
        det, obs = mem.compile_detector_sampler(seed=int(rng.integers(2**62))).sample(cfg["block_shots"],separate_observables=True)
        x.append(det)
        y.append(obs)
    batches = np.concatenate(x).reshape(-1,window,nominal.num_detectors)
    labels = np.concatenate(y).reshape(len(batches),window,nominal.num_observables)
    looks = len(batches)-1
    # Each policy only has one non-factory window. Score that intervention once per possible time.
    nominal_errors = []
    replacement_errors = []
    alarms = []
    for i,batch in enumerate(batches):
        nominal_errors.append(int(np.count_nonzero(factory.decode_batch(batch) != labels[i])))
        if i < looks:
            alarms.append(drift_alarm(batch,reference,len(ref),looks,cfg["alpha"]))
            graph = prior.matching(prior.estimate(batch),cfg["prior_fraction"])
            replacement_errors.append(int(np.count_nonzero(graph.decode_batch(batches[i+1]) != labels[i+1])))
    # Select using only alarms in temporal order. Force the final available fit when no alarm occurred.
    chosen = next((i for i,alarm in enumerate(alarms) if alarm),looks-1)
    counts = [sum(nominal_errors)-nominal_errors[i+1]+replacement_errors[i] for i in range(looks)]
    return dict(seed=seed, distance=d, factory_errors=nominal_errors, candidate_errors=counts,
                first_alarm_look=chosen+1 if any(alarms) else None, selected_look=chosen+1, selective_errors=counts[chosen])


def train(cfg):
    history = json.loads((ROOT/"data/suite_results.json").read_text())["trajectories"]
    result = {}
    for d in (3,5):
        cases = [r for r in history if (r["distance"],r["basis"],r["window"],r["scenario"])==(d,"Z",2000,"hotspot")]
        rows = [run_case(d,r["seed"],cfg) for r in cases]
        if any(r["factory_errors"] != old["errors"]["static"] for r,old in zip(rows,cases)):
            raise ValueError("Development streams disagree with the archived experiment")
        totals = np.sum([r["candidate_errors"] for r in rows],axis=0)
        result[str(d)] = dict(training_seeds=len(rows), totals=totals.tolist(), best_look=int(totals.argmin())+1)
    return result


def evaluate(cfg):
    settings = yaml.safe_load((ROOT/"configs/timing_confirmation.yaml").read_text())
    training = json.loads((ROOT/"data/timing_development.json").read_text())
    output = {}
    for panel in settings["panels"]:
        d, look = panel["distance"], training[str(panel["distance"])]["best_look"]
        rows = [run_case(d,seed,cfg) for seed in range(panel["first_seed"],panel["first_seed"]+panel["count"])]
        diffs = [(r["selective_errors"]-r["candidate_errors"][look-1])/12000 for r in rows]
        outcome = contrast(diffs)
        output[str(d)] = dict(fixed_look=look, records=rows, fixed_errors=sum(r["candidate_errors"][look-1] for r in rows),
                              selective_errors=sum(r["selective_errors"] for r in rows), paired=outcome)
    primary = output["3"]["paired"]
    return dict(primary_success=bool(primary["mean"] <= -settings["minimum_effect"] and primary["p"] < settings["alpha"]),
                provenance="independent seeded simulation of the follow-up timing hypothesis", panels=output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["train","evaluate"])
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    cfg = yaml.safe_load((ROOT/"configs/suite.yaml").read_text())
    result = train(cfg) if args.mode == "train" else evaluate(cfg)
    args.output.write_text(json.dumps(result,allow_nan=False,indent=2)+"\n", encoding="utf8",newline="\n")
