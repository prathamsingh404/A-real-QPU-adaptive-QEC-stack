"""Paired next-batch graph updates on independently calibrated memory experiments."""
from functools import lru_cache

import numpy as np
import pymatching
import stim
from scipy.stats import hypergeom

from adaptive_qec.controller.reweighting_gate import ConfidenceGatedReweightingController
from adaptive_qec.decoders.graph_fit import SpitzGraph

ARMS = ("static", "raw", "shrink", "rolling", "periodic", "selective", "legacy", "matched")


@lru_cache(maxsize=256)
def circuit(distance, basis, kind, rate, base, target=None):
    p = rate if kind == "uniform" else base
    nominal = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis.lower()}", distance=distance, rounds=distance,
        after_clifford_depolarization=p, after_reset_flip_probability=p,
        before_measure_flip_probability=p, before_round_data_depolarization=p,
    ).flattened()
    if kind in ("stationary", "uniform"):
        return nominal
    data = data_qubits(nominal)
    if kind == "hotspot" and target not in data:
        raise ValueError("Hotspot must target a data qubit")
    if kind not in ("hotspot", "bias"):
        raise ValueError(f"Unknown noise model {kind}")
    changed = stim.Circuit()
    for op in nominal:
        changed.append(op)
        if op.name == "TICK":
            if kind == "hotspot":
                changed.append("DEPOLARIZE1", [target], rate)
            else:
                # A local dephasing channel before each gate layer, including repeated rounds.
                changed.append("Z_ERROR", data, rate)
    return changed


def data_qubits(memory):
    readout = next(op for op in reversed(list(memory)) if op.name in ("M", "MX"))
    return [q.value for q in readout.targets_copy()]


def matching(memory):
    return pymatching.Matching.from_detector_error_model(memory.detector_error_model(decompose_errors=True))


def drift_alarm(samples, ref_counts, ref_n, looks, alpha):
    """Two-sided Fisher tails and Bonferroni across detectors and planned looks."""
    counts = samples.sum(axis=0, dtype=np.int64)
    totals = counts + ref_counts
    n = len(samples)
    lo = hypergeom.cdf(counts, ref_n + n, totals, n)
    hi = hypergeom.sf(counts - 1, ref_n + n, totals, n)
    return bool(np.minimum(lo, hi).min() <= alpha / (2 * len(counts) * looks))


class Policies:
    def __init__(self, factory, reference, windows, settings):
        self.factory = factory
        self.fit = SpitzGraph(factory)
        self.active = {k: factory for k in ARMS if k != "matched"}
        self.windows = windows
        self.cfg = settings
        self.ref_n = len(reference)
        self.ref_counts = reference.sum(axis=0, dtype=np.int64)
        self.legacy = ConfidenceGatedReweightingController(factory)
        self.legacy.reset_reference(new_reference=float(reference.mean()),
                                    new_reference_samples=reference.mean(axis=1))
        self.previous = None
        self.uses = 0
        self.periods = {max(1, round(windows * i / (settings["budget"] + 1)))
                        for i in range(1, settings["budget"] + 1)}

    def update(self, samples, completed):
        """Choose graphs for the *next* window. Labels and scenario metadata are inaccessible."""
        actions = {k: "stay" for k in self.active}
        if completed == self.windows:
            return actions
        estimate = self.fit.estimate(samples)
        blend = self.cfg["prior_fraction"]
        raw = self.fit.matching(estimate, 0)
        shrunk = self.fit.matching(estimate, blend)
        self.active["raw"], self.active["shrink"] = raw, shrunk
        actions["raw"] = actions["shrink"] = actions["rolling"] = "fit"
        self.active["rolling"] = shrunk if self.previous is None else self.fit.matching(
            self.fit.estimate(np.concatenate((self.previous, samples))), blend)
        self.previous = samples
        if completed in self.periods:
            self.active["periodic"] = shrunk
            actions["periodic"] = "fit"
        alarm = drift_alarm(samples, self.ref_counts, self.ref_n, self.windows - 1, self.cfg["alpha"])
        if alarm and self.active["selective"] is self.factory and self.uses < self.cfg["budget"]:
            self.active["selective"] = shrunk
            self.uses += 1
            actions["selective"] = "fit"
        elif not alarm and self.active["selective"] is not self.factory:
            self.active["selective"] = self.factory
            actions["selective"] = "reset"
        decision, self.active["legacy"], _ = self.legacy.evaluate_window(samples)
        actions["legacy"] = {"stay": "stay", "reweight": "fit", "reset": "reset"}[decision.value]
        return actions


def simulate_case(task):
    cfg, d, basis, seed, kind = task
    unit = cfg["block_shots"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, d, ord(basis), cfg["regimes"].index(kind)]))
    nominal = circuit(d, basis, "stationary", 0, cfg["base"])
    factory = matching(nominal)
    target = int(rng.choice(data_qubits(nominal)))
    start = int(rng.integers(12, 21))
    stop = start + int(rng.integers(16, 29))
    amplitude = float(rng.choice(cfg["amplitudes"][kind]))
    ref_seed = int(rng.integers(2**62))
    reference = nominal.compile_detector_sampler(seed=ref_seed).sample(cfg["reference_shots"])
    reference_matchers = {}
    batches, labels, matched_errors, levels = [], [], [], []
    for block in range(cfg["blocks"]):
        fraction = 0
        if kind != "stationary" and start <= block < stop:
            fraction = min(1, (block - start + 1) / 6, (stop - block) / 6) if kind == "uniform" else 1
        level = cfg["base"] + fraction * (amplitude - cfg["base"]) if kind == "uniform" else fraction * amplitude
        active_kind = kind if fraction else "stationary"
        mem = circuit(d, basis, active_kind, level, cfg["base"], target if active_kind == "hotspot" else None)
        batch_seed = int(rng.integers(2**62))
        x, y = mem.compile_detector_sampler(seed=batch_seed).sample(unit, separate_observables=True)
        if (active_kind, level) not in reference_matchers:
            reference_matchers[active_kind, level] = matching(mem)
        pred = reference_matchers[active_kind, level].decode_batch(x)
        matched_errors.append(int(np.count_nonzero(pred != y)))
        batches.append(x)
        labels.append(y)
        levels.append(float(level))
    shots, truth = np.concatenate(batches), np.concatenate(labels)
    rows = []
    for w in cfg["windows"]:
        n = len(shots) // w
        policies = Policies(factory, reference, n, cfg)
        by_arm = {k: [] for k in ARMS}
        decisions = []
        for i in range(n):
            x, y = shots[i*w:(i+1)*w], truth[i*w:(i+1)*w]
            decoded = {}
            for arm, graph in policies.active.items():
                if id(graph) not in decoded:
                    decoded[id(graph)] = int(np.count_nonzero(graph.decode_batch(x) != y))
                by_arm[arm].append(decoded[id(graph)])
            by_arm["matched"].append(sum(matched_errors[i*w//unit:(i+1)*w//unit]))
            decisions.append(policies.update(x, i+1))
        rows.append(dict(seed=seed, distance=d, basis=basis, scenario=kind, window=w,
                         shots=len(shots), onset_block=start if kind!="stationary" else None,
                         recovery_block=stop if kind!="stationary" else None,
                         physical_rates=levels, target=target if kind=="hotspot" else None,
                         errors=by_arm, actions=decisions))
    return rows
