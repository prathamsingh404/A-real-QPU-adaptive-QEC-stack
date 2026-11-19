"""Regenerate the complete simulation and measured-data replays, leaving released files intact."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from check_single_fit import compare
from make_claims import matches
from make_lock import ROOT, compute_sha256
from replay_public_memory import replay
from replay_alignment import replay as replay_alignment


def main():
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "simulation.json"
        subprocess.run([sys.executable, "experiments/run_suite.py", "--output", str(target),
                        "--manifest", str(Path(tmp) / "manifest.json")], cwd=ROOT, check=True)
        if compute_sha256(target) != compute_sha256(ROOT / "data/suite_results.json"):
            raise SystemExit("Simulation differs. Check the recorded Stim version and CPU instruction set.")
    for name, method in [("public_replay_results", replay), ("public_alignment_results", replay_alignment)]:
        if method() != json.loads((ROOT / f"data/{name}.json").read_text()):
            raise SystemExit(f"Replay differs: {name}")
    if not matches(compare(), json.loads((ROOT / "data/midpoint_results.json").read_text())):
        raise SystemExit("Single-fit sensitivity differs")
    subprocess.run([sys.executable, "scripts/make_claims.py", "--check"], cwd=ROOT, check=True)
    print("Simulation and both physical-data replays reproduced.")


if __name__ == "__main__":
    main()
