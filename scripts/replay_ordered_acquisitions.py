"""Score each Google 72Q acquisition using only past acquisitions for policy updates."""
import argparse
import json

import numpy as np
import stim

from analyze_correlated_study import ARMS, DATA, ROOT, read
from replay_public_memory import read_bits
from study_correlated_updates import compile_library, score


def check_model_fit():
    """Post hoc diagnostic on deployed graphs, using following-acquisition inputs."""
    released = read("ordered")
    diagnostic = {}
    for basis, result in released.items():
        base = ROOT / "data/ordered_acquisitions"
        folder = f"d3_at_q3_5/{basis}/r10"
        mem = stim.Circuit.from_file(base / "sample_13" / folder / "circuit_noisy_si1000.stim")
        library, _ = compile_library(str(mem))
        groups = {k: {"windows": 0, "excess_errors": 0} for k in ("higher", "lower_or_equal")}
        for i in range(1, 8):
            x = read_bits(base / f"sample_{13+i}" / folder / "detection_events.b8", 50000, mem.num_detectors)
            for offset, stream in enumerate(result["streams"]):
                model = stream["library_budget"]["actions"][i-1]
                if model == 0:
                    continue
                batch = x[offset::50]
                gain = (library.log_odds[model] - library.log_odds[0]) @ batch.sum(axis=0)
                gain += len(batch) * (library.log_quiet[model] - library.log_quiet[0])
                group = groups["higher" if gain > 0 else "lower_or_equal"]
                group["windows"] += 1
                group["excess_errors"] += stream["library_budget"]["errors"][i] - stream["factory"]["errors"][i]
        diagnostic[basis] = groups
    return diagnostic


def replay():
    settings = read("development")
    result = {}
    for basis in ("X", "Z"):
        detectors, labels = [], []
        harmony = {k: [] for k in ("si1000", "rl_optimized")}
        for sample in range(13, 21):
            folder = ROOT / f"data/ordered_acquisitions/sample_{sample}/d3_at_q3_5/{basis}/r10"
            metadata = json.loads((folder / "metadata.json").read_text())
            n = metadata["shots"]
            assert n == 50000
            mem = stim.Circuit.from_file(folder / "circuit_noisy_si1000.stim")
            if sample == 13:
                first = mem
            if mem.without_noise() != first.without_noise():
                raise ValueError(f"Circuit layout differs in {folder}")
            detectors.append(read_bits(folder / "detection_events.b8", n, mem.num_detectors))
            y = read_bits(folder / "obs_flips_actual.b8", n, 1)
            labels.append(y)
            for prior, errors in harmony.items():
                path = folder / "decoding_results" / f"harmony_decoder_with_{prior}_prior" / "obs_flips_predicted.b8"
                errors.append(int(np.count_nonzero(read_bits(path, n, 1) != y)))
        x, y = np.stack(detectors), np.stack(labels)
        streams = []
        for offset in range(50):
            arms = score(first, x[:,offset::50], y[:,offset::50], settings, offset)
            streams.append({k: {field: arms[k][field] for field in ("actions","errors")} for k in ARMS})
        result[basis] = dict(streams=streams, publisher=harmony)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--diagnostic", action="store_true", help="Check next-sample model fit on deployed windows")
    args = parser.parse_args()
    result = check_model_fit() if args.diagnostic else replay()
    target = "ordered_model_fit" if args.diagnostic else "ordered"
    if args.check:
        if result != read(target):
            raise SystemExit("Ordered acquisition replay differs")
        print(f"Reproduced {target}.")
    else:
        (DATA / f"{target}.json").write_text(json.dumps(result) + "\n", encoding="utf8", newline="\n")
