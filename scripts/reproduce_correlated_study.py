"""Re-run the frozen development and all held-out trials without changing released traces."""
from analyze_correlated_study import read
from study_correlated_updates import development, evaluate, physical
import json


def scientific(records):
    for record in records:
        for arm in record["arms"].values():
            del arm["times_ms"]
            arm.pop("setup_ms_per_circuit", None)
    return records


if __name__ == "__main__":
    settings = json.loads(json.dumps(development()))
    if settings != read("development"):
        raise SystemExit("Development choices or hindsight counts differ")
    for name,run in [("simulation", evaluate), ("physical", physical)]:
        actual = scientific(run(settings))
        if actual != scientific(read(name)):
            raise SystemExit(f"Scientific outcomes differ for {name}")
        print(f"Matched errors and actions for all {len(actual)} {name} trajectories.", flush=True)
