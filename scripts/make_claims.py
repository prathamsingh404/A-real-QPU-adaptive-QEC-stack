"""Derive the public results ledger from archived integer errors and controller actions."""
import argparse
import json
import math
from pathlib import Path

import yaml
from adaptive_qec.analysis.causal_results import summarize

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf8"))


def build():
    data = read("data/suite_results.json")
    cfg = yaml.safe_load((ROOT / "configs/suite.yaml").read_text())
    summary = summarize(data, cfg)
    primary = summary["primary"]
    rows = data["trajectories"]
    total = sum(r["shots"] for r in rows if r["window"] == cfg["windows"][0])
    lines = ["# Recalibration study results", "",
             f"The simulation scores {total:,} distinct shots. The two window sizes reuse each stream. "
             "Each decoder sees the same shots. Independent trajectories are the units of inference.", "",
             f"The primary mean difference is {100*primary['mean']:+.4f} percentage points. "
             f"The pointwise 95% interval is [{100*primary['ci95'][0]:+.4f}, {100*primary['ci95'][1]:+.4f}]. "
             f"The raw paired p-value is {primary['p']:.5f}, Holm-adjusted p is {summary['holm']['hotspot']:.5f}, "
             f"and the two-sided signed-rank sensitivity p is {primary['signed_rank_p']:.5f}. "
             f"The predefined success rule returned **{summary['primary_success']}**.", "",
             "## Simulated logical errors", "",
             "Fits count graph rebuilds for selective and periodic policies, respectively. "
             "The matched model knows the generating noise law.", "",
             "|Distance|Basis|Window|Regime|Seeds|Shots|Factory|Raw|Shrink|Rolling|Periodic|Selective|Historical|Matched|Fits S/P|",
             "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for p in summary["panels"]:
        cells = [p[k] for k in ("distance","basis","window","scenario","seeds","shots")]
        cells += [p["errors"][k] for k in ("static","raw","shrink","rolling","periodic","selective","legacy","matched")]
        cells.append(f"{p['fits']['selective']}/{p['fits']['periodic']}")
        lines.append("|" + "|".join(map(str, cells)) + "|")
    extra = read("data/midpoint_results.json")
    cases = extra["records"]
    lines += ["", "## Exploratory single-fit comparison", "",
              "These reuse the primary hotspot trajectories. Each arm rebuilds once per seed. The independent "
              "fixed schedule fits at the midpoint. The alarm schedule uses the original causal alarm-entry decision. "
              "Restoring the cached graph has a cost and requires no new fit.", "",
              "|Timing and retention|Total logical errors|", "|---|---:|"]
    for name in cases[0]["ablation_errors"]:
        count = sum(sum(r["ablation_errors"][name]) for r in cases)
        lines.append(f"|{name.replace('_',' ')}|{count}|")
    count = sum(sum(r["selective_errors"]) for r in cases)
    lines.append(f"|alarm with observed clearance|{count}|")
    lines += ["", "## Offline experimental replay", "",
              "Each basis contributes 44,000 held-out shots from one Google device placement. "
              "The acquisition timestamps are unknown. Alignment is a post hoc implementation of the first DGR stage. "
              "Publisher predictions are included in the released data.", "",
              "|Basis|Window|Factory|Always shrink|Periodic|Selective|DGR alignment|Publisher correlated|",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    alignment = {(r["basis"],r["window"]):sum(r["errors"]) for r in read("data/public_alignment_results.json")["records"]}
    for r in read("data/public_replay_results.json")["records"]:
        cells = [r["basis"],r["window"],*[sum(r["errors"][a]) for a in ("static","shrink","periodic","selective")],
                 alignment[r["basis"],r["window"]],sum(r["errors"]["publisher_correlated"])]
        lines.append("|" + "|".join(map(str,cells)) + "|")
    lines += ["", "Per-window errors, fit and reset decisions, host timing samples, and scored public files are in data/. "
              "The paper tables include the host timing distribution. The historical IBM summaries lack raw registers. "
              "The active study draws no IBM logical performance or feedback latency conclusion.", ""]
    return {"VALIDATION.md":"\n".join(lines),
            "data/analysis.json":json.dumps(summary, allow_nan=False, indent=2) + "\n"}


def matches(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(matches(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(matches(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, value in build().items():
        path = ROOT / name
        if args.check:
            stored = path.read_text(encoding="utf8")
            valid = matches(json.loads(stored), json.loads(value)) if name.endswith(".json") else stored == value
            if not valid:
                raise SystemExit(f"Stale generated output: {name}")
        else:
            path.write_text(value, encoding="utf8", newline="\n")
    print("Verified scientific ledger." if args.check else "Wrote scientific ledger.")


if __name__ == "__main__":
    main()
