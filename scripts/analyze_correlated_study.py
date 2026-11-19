"""Check the evaluation design and render the correlated-matching results from raw records."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
from replay_public_memory import read_bits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/correlated_study"
REPORT = ROOT / "reports/CORRELATED_STUDY.md"
REGIMES = ("stationary", "abrupt", "gradual", "recurring", "uniform", "bias")
ARMS = ("factory", "calendar", "always", "selective", "library", "library_calendar", "library_budget")
PAIRS = (("selective", "calendar"), ("library_budget", "library_calendar"))


def read(name):
    return json.loads((DATA / f"{name}.json").read_text())


def check(sim, physical):
    expected = {(d, r, 320000 + 1000*d + 24*i + j)
                for d, n in [(3, 24), (5, 8)] for i, r in enumerate(REGIMES) for j in range(n)}
    keys = [(r["distance"], r["regime"], r["seed"]) for r in sim]
    assert len(keys) == len(expected) and set(keys) == expected
    keys = [(r["patch"], r["basis"], r["file_start"]) for r in physical]
    expected = {(p, b, j*8000) for p in ("d3_0+0j", "d3_0+4j") for b in "XZ" for j in range(12)}
    assert len(keys) == len(expected) and set(keys) == expected
    for r in sim + physical:
        arms = r["arms"]
        assert set(arms) == set(ARMS)
        # Separate construction paths may use different graph tie-breaking across families.
        for family in [ARMS[:4], ARMS[4:]]:
            assert len({arms[name]["errors"][0] for name in family}) == 1
        for name, arm in arms.items():
            assert len(arm["errors"]) == len(arm["actions"]) == 8
            assert all(isinstance(e, int) and 0 <= e <= 1000 for e in arm["errors"])
            times = np.asarray(arm["times_ms"])
            assert times.shape == (8, 2) and np.isfinite(times).all() and (times >= 0).all()
            assert arm["actions"][-1] in ("factory", 0)
            if name in ("selective", "calendar", "library_budget", "library_calendar"):
                assert sum(a not in ("factory", 0) for a in arm["actions"]) <= 2


def losses(rows, arm):
    return np.array([sum(r["arms"][arm]["errors"]) for r in rows])


def times(rows, arm):
    return np.asarray([r["arms"][arm]["times_ms"] for r in rows])


def render():
    sim, physical, dev = read("simulation"), read("physical"), read("development")
    check(sim, physical)
    text = ["# Correlated matching under a two-update cap", "",
            "Seven policies scored identical shots. Commit `c2f80af` froze the code, choices, public inputs and internal protocol before evaluation. Negative paired differences favor selective timing. A shot is a complete memory experiment with one binary logical label.", "",
            "## Frozen primary results", "",
            "Distance three, Z memory, 24 independent trajectories in each of four regimes, 768,000 shots. Pointwise 95% Student intervals and two-sided paired t tests use trajectories as units. Holm adjustment covers both tests. The three success criteria were an adjusted p below .05, a reduction of at least 0.060 percentage points, and aggregate online time at most the matched calendar’s.", "",
            "| Policy vs calendar | Errors, policy / calendar | Difference in percentage points [95% CI] | Raw p | Holm p | Host time, policy / calendar (s) | Passed all three |",
            "|---|---:|---:|---:|---:|---:|---|"]
    primary = [r for r in sim if r["distance"] == 3 and r["regime"] in REGIMES[:4]]
    tests = []
    for a,b in PAIRS:
        differences = (losses(primary, a) - losses(primary, b)) * 100 / 8000
        t = stats.ttest_1samp(differences, 0)
        tests.append((a,b,differences.mean(),t.confidence_interval(),t.pvalue))
    order = np.argsort([test[-1] for test in tests])
    adjusted = np.zeros(2)
    adjusted[order] = np.minimum(1, np.maximum.accumulate([2*tests[order[0]][-1], tests[order[1]][-1]]))
    for (a,b,mean,ci,p),adj in zip(tests, adjusted):
        ta,tb = times(primary,a).sum()/1000, times(primary,b).sum()/1000
        success = adj < 0.05 and mean <= -0.06 and ta <= tb
        text.append(f"| {a} vs {b} | {losses(primary,a).sum():,} / {losses(primary,b).sum():,} | {mean:+.5f} [{ci.low:+.5f}, {ci.high:+.5f}] | {p:.5g} | {adj:.5g} | {ta:.3f} / {tb:.3f} | {'Yes' if success else 'No'} |")
    text.extend(["", "## Absolute error counts on every condition", "",
                 "Calendar and always fit full empirical priors. Selective chooses factory, local or full. The three library columns respectively allow all looks, two development-selected looks, or all looks with at most two uses. Model selection uses detector data from the preceding window. Uniform and bias are transfer checks. Physical rows each cover one released record in file order.", "",
                 "| Panel | Shots | " + " | ".join(ARMS) + " |",
                 "|---|" + "---:|" * (1 + len(ARMS))])
    def row(label, rows):
        counts = [f"{x:,}" for x in [len(rows)*8000, *(losses(rows,k).sum() for k in ARMS)]]
        return "| " + " | ".join([label, *counts]) + " |"
    for d in (3,5):
        for regime in REGIMES:
            text.append(row(f"d={d}, {regime}", [r for r in sim if r["distance"]==d and r["regime"]==regime]))
    for patch,basis in sorted({(r["patch"],r["basis"]) for r in physical}):
        text.append(row(f"Willow {patch}, {basis}", [r for r in physical if (r["patch"],r["basis"])==(patch,basis)]))
    text.append(row("Willow pooled, descriptive",physical))
    pathways = ("sparse_blossom_decoder_with_si1000_prior", "tesseract_decoder_with_si1000_prior",
                "tesseract_decoder_with_frequency_calibrated_prior", "alphaqubit2_decoder")
    text.extend(["", "## Released predictions on the same physical shots", "",
                 "This exploratory comparison reads the publisher’s predictions on exactly the first 96,000 shots of each record. Training and calibration independence from these shots is unverified. Runtime for these decoders was not measured. The sparse-blossom column uses the archive member name. The release README describes a correlated matching pathway under a different name.", "",
                 "| Record | Our budgeted library | Released sparse blossom | Tesseract SI1000 | Tesseract calibrated | AlphaQubit 2 |",
                 "|---|---:|---:|---:|---:|---:|"])
    total = np.zeros(5, dtype=int)
    for patch,basis in sorted({(r["patch"],r["basis"]) for r in physical}):
        directory = ROOT / "data/willow_timing" / patch / basis / "r010"
        metadata = json.loads((directory / "metadata.json").read_text())
        rows = [r for r in physical if (r["patch"], r["basis"]) == (patch,basis)]
        n = metadata["shots"]
        stop = len(rows)*8000
        actual = read_bits(directory / "obs_flips_actual.b8", n, 1)[:stop]
        counts = [losses(rows,"library_budget").sum()]
        for pathway in pathways:
            pred = read_bits(directory / "decoding_results" / pathway / "obs_flips_predicted.b8", n, 1)
            counts.append(np.count_nonzero(pred[:stop] != actual))
        total += counts
        text.append("| " + " | ".join([f"{patch}, {basis}", *(f"{c:,}" for c in counts)]) + " |")
    text.append("| Pooled | " + " | ".join(f"{c:,}" for c in total) + " |")
    physical_uses = sum(choice != 0 for r in physical for choice in r["arms"]["library_budget"]["actions"])
    text.append(f"\nThe budgeted library selected a hotspot model for {physical_uses} measured windows out of 336 possible next windows.")
    text.extend(["", "## Forward transfer between physical acquisitions", "",
                 "Google 72-qubit set 1 provides eight ordered acquisition samples per basis at one distance-three layout. Each decision scores a later sample. The 50 interleaved streams per basis share acquisition conditions. Counts are descriptive. Each column within a row uses the same 400,000 physical shots.", "",
                 "| Basis | " + " | ".join(ARMS) + " | Harmony SI1000 | Harmony optimized |",
                 "|---|" + "---:|" * (len(ARMS) + 2)])
    ordered = read("ordered")
    for basis in ("X", "Z"):
        streams = ordered[basis]["streams"]
        assert len(streams) == 50
        counts = [sum(sum(s[k]["errors"]) for s in streams) for k in ARMS]
        counts.extend(sum(ordered[basis]["publisher"][k]) for k in ("si1000", "rl_optimized"))
        text.append("| " + " | ".join([basis, *(f"{n:,}" for n in counts)]) + " |")
    text.append("\nThe common factory circuit is the first sample’s SI1000 model. The sources attribute temporal order at the acquisition level. The design makes no assumption about the chronology of rows within a sample. Both schedules and model strengths were frozen in simulation. The Harmony rows use released predictions and the training independence of optimized priors is unverified. [Exploratory protocol](ordered_acquisition_protocol.md) · [Per-stream counts and actions](../data/correlated_study/ordered.json) · [Source acquisition records](../data/ordered_acquisitions/ATTRIBUTION.md)")
    fit = read("ordered_model_fit")
    text.extend(["","### Did the selected model still fit one acquisition later?", "",
                 "Post hoc diagnostic restricted to active budgeted-library windows. We recomputed the label-free marginal composite log likelihood on the deployment window and compared the chosen graph with factory. Positive error differences are excess logical errors under the chosen graph.", "",
                 "| Next-window score vs factory | Active windows | Excess logical errors |", "|---|---:|---:|"])
    for key in ("higher", "lower_or_equal"):
        windows = sum(x[key]["windows"] for x in fit.values())
        errors = sum(x[key]["excess_errors"] for x in fit.values())
        text.append(f"| {key.replace('_', ' ')} | {windows} | {errors:+,} |")
    text.append("\nThis tests persistence of the selection score and its alignment with the logical objective. The grouped results describe these acquisitions. They do not identify the device fault responsible.")
    text.extend(["","## Measured work", "",
                 "The primary panel includes 768 windows per arm. Milliseconds per 1,000 shots are medians across windows. Times cover host decoding and selection or refitting. The library is shared between its three policies. First use may incur lazy initialization. Cold setup below measures graph construction separately. Acquisition, transport, scoring and device feedback lie outside these clocks.","",
                 "| Policy | Active next windows | Decode ms | Decision/build ms | Total ms, 95th percentile |",
                 "|---|---:|---:|---:|---:|"])
    for a in ARMS:
        t = times(primary,a).reshape(-1,2)
        actions = Counter(k for r in primary for k in r["arms"][a]["actions"])
        active = sum(n for k,n in actions.items() if k not in ("factory",0))
        text.append(f"| {a} | {active} | {np.median(t[:,0]):.3f} | {np.median(t[:,1]):.3f} | {np.percentile(t.sum(axis=1),95):.3f} |")
    for d in (3,5):
        setup = next(r for r in sim if r["distance"]==d)["arms"]["library"]["setup_ms_per_circuit"]
        text.append(f"\nSimulated distance {d} required {d*d+1} graphs and {setup:.1f} ms of cold library construction.")
    setup = [r["arms"]["library"]["setup_ms_per_circuit"] for r in physical if r["file_start"]==0]
    text.append(f"\nThe four physical libraries required {min(setup):.1f}–{max(setup):.1f} ms each.")
    ceiling = sum(sum(v) for v in dev["hindsight_gains"].values())
    text.extend(["","## What the design can resolve","",
                 f"On 256,000 development shots, a label-informed hindsight selector could have saved at most {ceiling} errors against factory among the tested empirical estimates with two single-window uses. The practical margin would require 154. This ceiling covers the enumerated candidates and traces only. Development selected empirical calendar looks 1 and 7, library looks 2 and 3, and penalty 32 (one-based looks).",
                 "", "The library knows that a candidate fault has the form of depolarization on a data qubit at each circuit tick. The simulator generates the local disturbance from that same family with different strength, timing and location. Improvement under this match is evidence about model selection given a fault family. The physical records cover two neighboring patches, two bases, ten cycles and a single calibration category. Consecutive file segments share a source and have no verified timestamps or drift labels. Their errors are descriptive. A physical timing test requires repeated, chronologically documented episodes and an experimental unit that reflects the acquisition design.",
                 "", "The cap limits active windows per episode. Schedules can inspect different numbers of windows. Cost is an observed criterion on one CPU host. The protocol supplies neither an enforced device deadline nor interleaved hardware acquisition.", "",
                 "[Protocol](correlated_protocol.md) · [Simulation traces](../data/correlated_study/simulation.json) · [Physical traces](../data/correlated_study/physical.json) · [Data attribution](../data/willow_timing/ATTRIBUTION.md)"])
    return "\n".join(text) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = render()
    if args.check:
        if REPORT.read_text(encoding="utf8") != report:
            raise SystemExit("Correlated study report differs from regenerated results")
    else:
        REPORT.write_text(report, encoding="utf8", newline="\n")
