"""Run the versioned causal recalibration benchmark from the repository root."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml

from adaptive_qec.experiments.causal_reweighting import simulate_case

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf8", newline="\n")
    temp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT/"configs/suite.yaml")
    ap.add_argument("--output", type=Path, default=ROOT/"data/suite_results.json")
    ap.add_argument("--manifest", type=Path, default=ROOT/"data/manifest.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.pilot and (args.output.resolve() == ROOT/"data/suite_results.json" or
                       args.manifest.resolve() == ROOT/"data/manifest.json"):
        ap.error("Supply separate --output and --manifest paths for pilot runs")
    panels = [dict(distance=3, basis="Z", first_seed=70000, count=1)] if args.pilot else cfg["panels"]
    tasks = [(cfg, p["distance"], p["basis"], seed, kind) for p in panels
             for seed in range(p["first_seed"], p["first_seed"]+p["count"]) for kind in cfg["regimes"]]
    start = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, rs in enumerate(pool.map(simulate_case, tasks), 1):
            rows.extend(rs)
            if n % 10 == 0 or n == len(tasks):
                print(f"{n}/{len(tasks)} streams", flush=True)
    save(args.output, dict(metadata=dict(schema=3, provenance="simulation", pilot=args.pilot), trajectories=rows))
    paths = [Path(__file__), ROOT/"src/adaptive_qec/experiments/causal_reweighting.py",
             ROOT/"src/adaptive_qec/decoders/graph_fit.py", ROOT/"src/adaptive_qec/decoders/dem_calibration.py",
             ROOT/"src/adaptive_qec/controller/reweighting_gate.py", ROOT/"PREREG.md"]
    save(args.manifest, dict(provenance="simulation", git_commit=subprocess.check_output(
        ["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
        config_sha256=digest(args.config), timestamp_utc=datetime.now(timezone.utc).isoformat(),
        command_line=subprocess.list2cmdline(sys.argv), execution_duration_sec=time.perf_counter()-start,
        system={"platform":platform.platform(), "python_version":platform.python_version(),
                **{k+"_version":metadata.version(k) for k in ("stim","pymatching","numpy","scipy")}},
        results_file=dict(total_records=len(rows), sha256=digest(args.output))))


if __name__ == "__main__":
    main()
