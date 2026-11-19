"""Train on independent simulated episodes and evaluate frozen policies on simulation and physical records."""
from functools import lru_cache
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import stim
from adaptive_qec.decoders.fault_library import FaultLibrary
from adaptive_qec.decoders.correlated_updates import CorrelatedPrior
from adaptive_qec.experiments.causal_reweighting import circuit, data_qubits
from replay_public_memory import read_bits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/correlated_study"
REGIMES = ["stationary","abrupt","gradual","recurring"]
W = 1000


def simulation(seed, regime, d=3):
    rng = np.random.default_rng([seed,(REGIMES+["uniform","bias"]).index(regime)])
    nominal = circuit(d,"Z","stationary",0,0.004)
    q = int(rng.choice(data_qubits(nominal)))
    onset = rng.integers(4,14)
    batches, labels = [], []
    for i in range(32):
        t = i-onset
        strength = 0
        if regime == "abrupt" and 0 <= t < 10:
            strength = 0.035
        if regime == "gradual" and 0 <= t < 12:
            strength = (6-abs(t-5.5))/6 * 0.04
        if regime == "recurring" and (0 <= t < 4 or 9 <= t < 13):
            strength = 0.04
        mem = circuit(d,"Z","hotspot",float(strength),0.004,q) if strength else nominal
        if regime == "uniform" and 0 <= t < 10:
            mem = circuit(d,"Z","uniform",0.009,0.004)
        if regime == "bias" and 0 <= t < 10:
            mem = circuit(d,"Z","bias",0.004,0.004)
        x,y = mem.compile_detector_sampler(seed=int(rng.integers(2**61))).sample(250,separate_observables=True)
        batches.append(x)
        labels.append(y)
    return nominal,np.concatenate(batches).reshape(8,W,-1),np.concatenate(labels).reshape(8,W,-1)


def run_arm(prior, x, y, name, parameter):
    factory = prior.graph()
    graph = factory
    decisions, errors, times = [], [], []
    remaining = 2
    for i, batch in enumerate(x):
        start = perf_counter_ns()
        prediction = graph.decode_batch(batch,enable_correlations=True)
        decoded = perf_counter_ns()
        graph = factory
        action = "factory"
        if i+1 < len(x):
            if name == "always" or (name == "calendar" and i in parameter):
                graph = prior.graph(prior.estimate(batch,"full"))
                action = "full"
            elif name == "selective" and remaining:
                action, a = prior.choose(batch,parameter)
                if a is not None:
                    graph = prior.graph(a)
                    remaining -= 1
        finished = perf_counter_ns()
        errors.append(int(np.count_nonzero(prediction!=y[i])))
        decisions.append(action)
        times.append([(decoded-start)/1e6,(finished-decoded)/1e6])
    return dict(errors=errors,actions=decisions,times_ms=times)


def best_two(gains):
    return sum(sorted([max(0,g) for g in gains],reverse=True)[:2])


@lru_cache(maxsize=8)
def compile_library(circuit_text):
    start = perf_counter_ns()
    library = FaultLibrary(stim.Circuit(circuit_text))
    return library,(perf_counter_ns()-start)/1e6


def library_arm(library,x,y,allowed,cap=None):
    choice = 0
    uses = 0
    errors, actions, times = [], [], []
    for i,batch in enumerate(x):
        start = perf_counter_ns()
        pred = library.graphs[choice].decode_batch(batch,enable_correlations=True)
        end = perf_counter_ns()
        choice = library.select(batch) if i in allowed and i+1<len(x) and (cap is None or uses<cap) else 0
        uses += int(choice != 0)
        done = perf_counter_ns()
        errors.append(int(np.count_nonzero(pred != y[i])))
        actions.append(choice)
        times.append([(end-start)/1e6,(done-end)/1e6])
    return dict(errors=errors,actions=actions,times_ms=times)


def development():
    scores = {k:0 for k in (1,8,32)}
    calendar = np.zeros(7,dtype=int)
    lib_gains = np.zeros(7,dtype=int)
    ceilings = {}
    for regime in REGIMES:
        ceilings[regime] = []
        for seed in range(310000,310008):
            mem,x,y = simulation(seed,regime)
            prior = CorrelatedPrior(mem)
            base = run_arm(prior,x,y,"factory",None)["errors"]
            lib,_ = compile_library(str(mem))
            lib_errors = library_arm(lib,x,y,range(7))["errors"]
            lib_gains += np.subtract(base[1:],lib_errors[1:])
            gains = []
            for i in range(7):
                outcomes = {}
                for action in ["local","full"]:
                    # Same candidate sample sizes as the deployed schedules.
                    for training in [x[i][:W//2],x[i]]:
                        graph = prior.graph(prior.estimate(training,action))
                        pred = graph.decode_batch(x[i+1],enable_correlations=True)
                        outcomes[action,len(training)] = int(np.count_nonzero(pred!=y[i+1]))
                calendar[i] += base[i+1]-outcomes["full",W]
                gains.append(base[i+1]-min(outcomes.values()))
            ceilings[regime].append(best_two(gains))
            for penalty in scores:
                scores[penalty] += sum(run_arm(prior,x,y,"selective",penalty)["errors"])
    chosen = sorted(range(7),key=lambda i:(-calendar[i],i))[:2]
    return dict(library_calendar=sorted(range(7),key=lambda i:(-lib_gains[i],i))[:2], library_gains=lib_gains.tolist(), penalty=min(scores,key=scores.get),calendar=sorted(chosen),
                development_errors=scores,calendar_gains=calendar.tolist(),hindsight_gains=ceilings)


def score(mem,x,y,settings,seed):
    prior = CorrelatedPrior(mem)
    arms = {"factory":None,"calendar":settings["calendar"],"always":None,"selective":settings["penalty"]}
    order = np.random.default_rng(seed).permutation(list(arms))
    results = {k:run_arm(prior,x,y,k,arms[k]) for k in order}
    lib, setup = compile_library(str(mem))
    schedules = {"library":range(7),"library_budget":range(7),"library_calendar":settings["library_calendar"]}
    for k in np.random.default_rng(seed+1).permutation(list(schedules)):
        results[k] = library_arm(lib,x,y,schedules[k],2 if k=="library_budget" else None)
        results[k]["setup_ms_per_circuit"] = setup
    return results


def evaluate(settings):
    records = []
    for d,count in [(3,24),(5,8)]:
        for r,regime in enumerate(REGIMES+["uniform","bias"]):
            for j in range(count):
                seed = 320000+1000*d+24*r+j
                records.append(dict(regime=regime,distance=d,seed=seed,
                                    arms=score(*simulation(seed,regime,d),settings,seed)))
    return records


def physical(settings):
    records = []
    for f in sorted((ROOT/"data/willow_timing").glob("*/*/r010/metadata.json")):
        info = json.loads(f.read_text())
        mem = stim.Circuit.from_file(f.parent/"circuit_noisy_si1000.stim")
        n = info["shots"]
        x = read_bits(f.parent/"detection_events.b8",n,mem.num_detectors)
        y = read_bits(f.parent/"obs_flips_actual.b8",n,mem.num_observables)
        # Apply the eight-window policy independently to consecutive complete file segments.
        for j in range(n//(8*W)):
            start = j*8*W
            sl = slice(start,start+8*W)
            records.append(dict(patch=f.parents[2].name,basis=info["basis"],file_start=start,
                                arms=score(mem,x[sl].reshape(8,W,-1),y[sl].reshape(8,W,-1),settings,j)))
    if len({(r["patch"],r["basis"]) for r in records})!=4:
        raise ValueError("Expected four complete Willow records")
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["development","simulation","physical"])
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.mode == "development":
        result = development()
    else:
        settings = json.loads((OUT/"development.json").read_text())
        result = evaluate(settings) if a.mode=="simulation" else physical(settings)
    destination = OUT/(a.mode+".json")
    tmp = destination.with_suffix(".tmp")
    tmp.write_text(json.dumps(result,allow_nan=False)+"\n",encoding="utf8",newline="\n")
    tmp.replace(destination)
