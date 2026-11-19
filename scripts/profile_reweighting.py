"""Measure single-policy host timings with real detector arrays. Run serially."""
import json
import platform
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import yaml

from adaptive_qec.decoders.graph_fit import SpitzGraph
from adaptive_qec.experiments.causal_reweighting import circuit, data_qubits, drift_alarm, matching

ROOT = Path(__file__).resolve().parents[1]


def profile():
    cfg = yaml.safe_load((ROOT/"configs/suite.yaml").read_text())
    rows = []
    for d in (3, 5, 7):
        mem = circuit(d, "Z", "stationary", 0, cfg["base"])
        factory = matching(mem)
        fit = SpitzGraph(factory)
        target = data_qubits(mem)[d*d//2]
        hot = circuit(d, "Z", "hotspot", 0.04, cfg["base"], target)
        sampler = hot.compile_detector_sampler(seed=70003)
        clean = mem.compile_detector_sampler(seed=70004)
        ref = clean.sample(cfg["reference_shots"])
        counts = ref.sum(axis=0, dtype=np.int64)
        for w in cfg["windows"]:
            for action, source, previous in (("stay", clean, False), ("fit", sampler, False), ("reset", clean, True)):
                timings = []
                for rep in range(33):
                    x = source.sample(w)
                    previous_graph = fit.matching(fit.estimate(sampler.sample(w)), cfg["prior_fraction"]) if previous else factory
                    t0 = perf_counter_ns()
                    previous_graph.decode_batch(x)
                    t1 = perf_counter_ns()
                    alarm = drift_alarm(x,counts,len(ref),cfg["blocks"]*cfg["block_shots"]//w-1,cfg["alpha"])
                    t2 = perf_counter_ns()
                    actual = "stay"
                    if alarm and not previous:
                        graph = fit.matching(fit.estimate(x), cfg["prior_fraction"])
                        actual = "fit"
                    elif not alarm and previous:
                        graph = factory
                        actual = "reset"
                    else:
                        graph = previous_graph
                    t3 = perf_counter_ns()
                    if rep >= 3:
                        timings.append(dict(action=actual,decode_us=(t1-t0)/1000,
                                            monitor_us=(t2-t1)/1000,update_us=(t3-t2)/1000,
                                            total_us=(t3-t0)/1000))
                    del graph
                rows.append(dict(distance=d,window=w,requested_path=action,samples=timings))
    return dict(provenance="host microbenchmark", platform=platform.platform(),
                cpu=platform.processor(), clock="perf_counter_ns", units="microseconds per complete window",
                warmup_per_case=3, records=rows)


if __name__ == "__main__":
    (ROOT/"data/host_timings.json").write_text(json.dumps(profile(),allow_nan=False)+"\n")
