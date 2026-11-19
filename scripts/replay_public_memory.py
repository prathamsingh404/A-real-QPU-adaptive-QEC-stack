"""Offline file-order replay of Google's 105-qubit memory records (Zenodo 13273331)."""
import json
from pathlib import Path

import numpy as np
import stim
import yaml

from adaptive_qec.experiments.causal_reweighting import Policies, matching

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data/public_replay/google_105Q_surface_code_d3_d5_d7/d3_at_q4_5"


def read_bits(path, n, width):
    raw = np.fromfile(path, dtype=np.uint8).reshape(n, (width+7)//8)
    return np.unpackbits(raw, axis=1, count=width, bitorder="little").astype(bool)


def replay():
    cfg = yaml.safe_load((ROOT/"configs/suite.yaml").read_text())
    rows = []
    for basis in ("X", "Z"):
        folder = DATA/basis/"r13"
        n = json.loads((folder/"metadata.json").read_text())["shots"]
        mem = stim.Circuit.from_file(folder/"circuit_noisy_si1000.stim")
        factory = matching(mem)
        x = read_bits(folder/"detection_events.b8", n, mem.num_detectors)
        y = read_bits(folder/"obs_flips_actual.b8", n, mem.num_observables)
        publisher = read_bits(folder/"decoding_results/correlated_matching_decoder_with_si1000_prior/obs_flips_predicted.b8",
                              n, mem.num_observables)
        ref = cfg["reference_shots"]
        for w in cfg["windows"]:
            assert (n-ref) % w == 0
            windows = (n-ref)//w
            policy = Policies(factory, x[:ref], windows, cfg)
            errors = {arm: [] for arm in [*policy.active, "publisher_correlated"]}
            actions = []
            for i, start in enumerate(range(ref,n,w), 1):
                batch = x[start:start+w]
                labels = y[start:start+w]
                for arm, graph in policy.active.items():
                    errors[arm].append(int(np.count_nonzero(graph.decode_batch(batch) != labels)))
                errors["publisher_correlated"].append(int(np.count_nonzero(publisher[start:start+w] != labels)))
                actions.append(policy.update(batch, i))
            rows.append(dict(basis=basis, window=w, shots=n-ref, calibration_shots=ref, errors=errors, actions=actions))
    return dict(provenance="offline replay of public physical measurements", source="https://doi.org/10.5281/zenodo.13273331",
                ordered_by="released file row; acquisition timing unverified", records=rows)


if __name__ == "__main__":
    result = replay()
    (ROOT/"data/public_replay_results.json").write_text(json.dumps(result, allow_nan=False)+"\n")
    for row in result["records"]:
        print(row["basis"], row["window"], {arm:sum(v) for arm,v in row["errors"].items()})
