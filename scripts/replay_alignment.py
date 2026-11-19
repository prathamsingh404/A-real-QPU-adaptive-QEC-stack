"""Exploratory implementation of Wang et al. DGR Section IV-A on the bundled Google records.

Count predicted correction-chain edge occurrences in the preceding completed window.
Use MWPM log odds and the same finite-weight clipping limits as the parity estimator.
The paper's second-stage conditional correlation reweighter is outside this comparator.
"""
import json
from pathlib import Path

import numpy as np
import stim
import yaml

from adaptive_qec.decoders.graph_fit import SpitzGraph
from adaptive_qec.experiments.causal_reweighting import matching
if __package__:
    from .replay_public_memory import DATA, ROOT, read_bits
else:
    from replay_public_memory import DATA, ROOT, read_bits


def key(u, v):
    return (u, -1) if v is None or v == -1 else tuple(sorted((u, v)))


def trace(graph, topology, shots):
    indices = {key(u, v): i for i, (u, v, _) in enumerate(topology.edges)}
    counts = np.zeros(len(indices), dtype=int)
    for sample in shots:
        for u, v in graph.decode_to_edges_array(sample):
            counts[indices[key(u, v)]] += 1
    return np.clip(counts / len(shots), 1e-4, 0.49)


def replay():
    cfg = yaml.safe_load((ROOT / "configs/suite.yaml").read_text())
    records = []
    for basis in ("X", "Z"):
        folder = DATA / basis / "r13"
        shots = json.loads((folder / "metadata.json").read_text())["shots"]
        mem = stim.Circuit.from_file(folder / "circuit_noisy_si1000.stim")
        factory = matching(mem)
        topology = SpitzGraph(factory)
        syndromes = read_bits(folder / "detection_events.b8", shots, mem.num_detectors)
        truth = read_bits(folder / "obs_flips_actual.b8", shots, mem.num_observables)
        for window in cfg["windows"]:
            graph = factory
            errors = []
            for i in range(cfg["reference_shots"], shots, window):
                x = syndromes[i:i+window]
                predictions = graph.decode_batch(x)
                errors.append(int(np.count_nonzero(predictions != truth[i:i+window])))
                if i + window < shots:
                    graph = topology.matching(trace(graph, topology, x), prior_fraction=0)
            records.append(dict(basis=basis, window=window, errors=errors))
    return dict(stage="exploratory after primary analysis", method="DGR alignment-only implementation",
                source="https://arxiv.org/html/2311.16214v3", records=records)


if __name__ == "__main__":
    result = replay()
    (ROOT / "data/public_alignment_results.json").write_text(json.dumps(result) + "\n", encoding="utf8")
    print([(r["basis"], r["window"], sum(r["errors"])) for r in result["records"]])
