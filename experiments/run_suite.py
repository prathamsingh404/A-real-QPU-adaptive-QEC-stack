"""
Reproducible Confirmatory Simulation Pipeline for Adaptive QEC.

Executes the formal benchmark suite specified in configs/suite.yaml across
held-out seeds and records all raw trajectory-level results.

Output:
  - data/suite_results.json: Raw trajectory-level records and aggregated endpoints.
  - data/manifest.json: Cryptographic manifest capturing commit, dependencies,
    config hash, timestamps, and execution metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pymatching
import scipy.stats as stats
import stim
import yaml

from adaptive_qec.controller.reweighting_gate import (
    ConfidenceGatedReweightingController,
    GateDecision,
)
from adaptive_qec.decoders.dem_calibration import calibrate_spitz_matching

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("RunSuite")


def get_git_commit_hash() -> str:
    """Retrieve the current Git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def build_circuit(
    distance: int,
    rounds: int,
    p_base: float,
    drift_type: str,
    drift_p: float,
    hotspot_qubit: Optional[int] = None,
) -> stim.Circuit:
    """Construct surface code circuit with specified physical noise dynamics."""
    # Base circuit below threshold
    circuit_base = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p_base,
        after_reset_flip_probability=p_base,
        before_measure_flip_probability=p_base,
        before_round_data_depolarization=p_base,
    )

    if drift_type == "stationary":
        return circuit_base

    if drift_type == "uniform_ramp":
        # Global uniform drift: all operations scaled to drift_p
        return stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=drift_p,
            after_reset_flip_probability=drift_p,
            before_measure_flip_probability=drift_p,
            before_round_data_depolarization=drift_p,
        )

    if drift_type == "localized_hotspot":
        # Elevated depolarizing noise on a single data transmon
        if hotspot_qubit is None:
            hotspot_qubit = 10 if distance == 3 else 27
        lines = []
        for line in str(circuit_base).splitlines():
            lines.append(line)
            if line.startswith("TICK"):
                lines.append(f"DEPOLARIZE1({drift_p}) {hotspot_qubit}")
        return stim.Circuit("\n".join(lines))

    if drift_type == "structural_anisotropy":
        # Asymmetric noise: severe dephasing (Z_ERROR) vs bit-flip (X_ERROR)
        lines = []
        for line in str(circuit_base).splitlines():
            lines.append(line)
            if line.startswith("TICK"):
                lines.append(f"Z_ERROR({drift_p})")
        return stim.Circuit("\n".join(lines))

    return circuit_base


def run_trajectory(
    seed: int,
    distance: int,
    scenario: dict[str, Any],
    window_size: int,
    total_shots: int,
    p_base: float,
) -> dict[str, Any]:
    """Run one trajectory across the 4 arms on identical syndromes."""
    rounds = distance
    drift_type = scenario["drift_type"]
    drift_p = scenario["drift_p"]
    hotspot_qubit = scenario.get(f"hotspot_qubit_d{distance}")

    # 1. Base Factory Circuit & DEM
    circuit_base = build_circuit(distance, rounds, p_base, "stationary", p_base)
    dem_base = circuit_base.detector_error_model(decompose_errors=True)
    matcher_factory = pymatching.Matching.from_detector_error_model(dem_base)

    # 2. Drifted Circuit & Oracle DEM
    circuit_drifted = build_circuit(distance, rounds, p_base, drift_type, drift_p, hotspot_qubit)
    dem_oracle = circuit_drifted.detector_error_model(decompose_errors=True)
    matcher_oracle = pymatching.Matching.from_detector_error_model(dem_oracle)

    # 3. Sample exact identical stream
    sampler = circuit_drifted.compile_detector_sampler(seed=seed)
    det, obs = sampler.sample(shots=total_shots, separate_observables=True)
    obs = obs.flatten()

    num_windows = total_shots // window_size
    arms = ["static_mwpm", "naive_reweighted_mwpm", "cg_ccads", "oracle_mwpm"]
    arm_errors = {arm: 0 for arm in arms}
    arm_switches = {arm: 0 for arm in arms}
    arm_stays = {arm: 0 for arm in arms}
    arm_rollbacks = {arm: 0 for arm in arms}

    # Tracking states
    cg_controller = ConfidenceGatedReweightingController(
        base_matching=matcher_factory,
        alpha_significance=0.05,
        drift_z_threshold=2.5,
        adaptation_tax=0.003,
        dead_zone_epsilon=0.002,
    )
    naive_matching = matcher_factory

    for w_idx in range(num_windows):
        start = w_idx * window_size
        end = start + window_size
        w_det = det[start:end]
        w_obs = obs[start:end]

        # --- Arm 0: Static Factory MWPM ---
        corr_static = matcher_factory.decode_batch(w_det).flatten()
        arm_errors["static_mwpm"] += int(np.sum(corr_static != w_obs))
        arm_stays["static_mwpm"] += 1

        # --- Arm 1: Naive Reweighted MWPM (Spitz et al.) ---
        corr_naive = naive_matching.decode_batch(w_det).flatten()
        arm_errors["naive_reweighted_mwpm"] += int(np.sum(corr_naive != w_obs))
        if w_idx > 0:
            arm_switches["naive_reweighted_mwpm"] += 1
        # Update naive weights for NEXT window from current window syndromes
        naive_matching = calibrate_spitz_matching(matcher_factory, w_det)

        # --- Arm 2: CG-CCADS (Label-Free Confidence-Gated) ---
        decision, active_m, stats_obj = cg_controller.evaluate_window(w_det)
        corr_cg = active_m.decode_batch(w_det).flatten()
        arm_errors["cg_ccads"] += int(np.sum(corr_cg != w_obs))
        if decision == GateDecision.REWEIGHT:
            arm_switches["cg_ccads"] += 1
        elif decision == GateDecision.RESET:
            arm_rollbacks["cg_ccads"] += 1
        else:
            arm_stays["cg_ccads"] += 1

        # --- Arm 3: Oracle Matched MWPM ---
        corr_oracle = matcher_oracle.decode_batch(w_det).flatten()
        arm_errors["oracle_mwpm"] += int(np.sum(corr_oracle != w_obs))
        arm_stays["oracle_mwpm"] += 1

    evaluated_shots = num_windows * window_size
    results: dict[str, Any] = {
        "seed": seed,
        "distance": distance,
        "scenario_id": scenario["id"],
        "window_size": window_size,
        "total_shots": evaluated_shots,
        "arms": {},
    }

    for arm in arms:
        ler = arm_errors[arm] / evaluated_shots
        results["arms"][arm] = {
            "errors": arm_errors[arm],
            "ler": float(ler),
            "switches": arm_switches[arm],
            "stays": arm_stays[arm],
            "rollbacks": arm_rollbacks[arm],
        }

    return results


def run_suite(
    config_path: Path,
    seed_range: Tuple[int, int],
    output_path: Path,
    manifest_path: Path,
) -> None:
    """Execute complete confirmatory simulation suite."""
    start_time = time.time()
    logger.info("Loading config from %s", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config_text = f.read()
    config = yaml.safe_load(config_text)
    config_hash = hashlib.sha256(config_text.encode("utf-8")).hexdigest()

    seeds = list(range(seed_range[0], seed_range[1]))
    distances = config["noise_regime"]["distances"]
    p_base = config["noise_regime"]["base_physical_error"]
    scenarios = config["scenarios"]
    window_regimes = [w["window_size"] for w in config["window_regimes"]]
    total_shots = config.get("trajectory", {}).get("total_shots", 10000)

    logger.info(
        "Starting suite execution: %d seeds (%d to %d), distances=%s, %d scenarios, %d window sizes",
        len(seeds),
        seeds[0],
        seeds[-1],
        distances,
        len(scenarios),
        len(window_regimes),
    )

    all_records: List[dict[str, Any]] = []
    total_runs = len(seeds) * len(distances) * len(scenarios) * len(window_regimes)
    run_idx = 0

    for seed in seeds:
        for dist in distances:
            for sc in scenarios:
                for w_size in window_regimes:
                    run_idx += 1
                    rec = run_trajectory(
                        seed=seed,
                        distance=dist,
                        scenario=sc,
                        window_size=w_size,
                        total_shots=total_shots,
                        p_base=p_base,
                    )
                    all_records.append(rec)
                    if run_idx % 25 == 0 or run_idx == total_runs:
                        logger.info("Progress: %d / %d runs completed (%.1f%%)", run_idx, total_runs, run_idx / total_runs * 100)

    elapsed_sec = time.time() - start_time
    logger.info("Simulations completed in %.2f seconds.", elapsed_sec)

    # --- Statistical Aggregation Across Seeds ---
    aggregates: dict[str, Any] = {}
    for dist in distances:
        aggregates[f"d{dist}"] = {}
        for sc in scenarios:
            sc_id = sc["id"]
            aggregates[f"d{dist}"][sc_id] = {}
            for w_size in window_regimes:
                # Filter matching records
                matching = [
                    r for r in all_records
                    if r["distance"] == dist and r["scenario_id"] == sc_id and r["window_size"] == w_size
                ]
                n_seeds = len(matching)
                if n_seeds == 0:
                    continue

                arm_data: dict[str, Any] = {}
                for arm_key in ["static_mwpm", "naive_reweighted_mwpm", "cg_ccads", "oracle_mwpm"]:
                    lers = np.array([r["arms"][arm_key]["ler"] for r in matching])
                    switches = np.array([r["arms"][arm_key]["switches"] for r in matching])
                    stays = np.array([r["arms"][arm_key]["stays"] for r in matching])
                    rollbacks = np.array([r["arms"][arm_key]["rollbacks"] for r in matching])
                    arm_data[arm_key] = {
                        "mean_ler": float(np.mean(lers)),
                        "std_ler": float(np.std(lers, ddof=1)) if n_seeds > 1 else 0.0,
                        "mean_switches": float(np.mean(switches)),
                        "mean_stays": float(np.mean(stays)),
                        "mean_rollbacks": float(np.mean(rollbacks)),
                    }

                # Paired delta analysis: CG-CCADS vs Static MWPM
                static_lers = np.array([r["arms"]["static_mwpm"]["ler"] for r in matching])
                cg_lers = np.array([r["arms"]["cg_ccads"]["ler"] for r in matching])
                naive_lers = np.array([r["arms"]["naive_reweighted_mwpm"]["ler"] for r in matching])

                delta_cg = cg_lers - static_lers
                delta_naive = naive_lers - static_lers

                mean_delta_cg = float(np.mean(delta_cg))
                std_delta_cg = float(np.std(delta_cg, ddof=1)) if n_seeds > 1 else 1e-6
                t_stat_cg, p_val_cg = stats.ttest_1samp(delta_cg, 0.0) if n_seeds > 1 else (0.0, 1.0)

                # 95% Confidence Interval
                ci_margin_cg = float(stats.t.ppf(0.975, df=max(n_seeds - 1, 1)) * (std_delta_cg / np.sqrt(n_seeds)))

                mean_delta_naive = float(np.mean(delta_naive))

                aggregates[f"d{dist}"][sc_id][f"w{w_size}"] = {
                    "n_seeds": n_seeds,
                    "arms": arm_data,
                    "paired_analysis": {
                        "cg_vs_static": {
                            "mean_delta_ler": mean_delta_cg,
                            "relative_change_pct": float(mean_delta_cg / max(np.mean(static_lers), 1e-9) * 100),
                            "std_delta_ler": std_delta_cg,
                            "ci_95": [float(mean_delta_cg - ci_margin_cg), float(mean_delta_cg + ci_margin_cg)],
                            "t_statistic": float(t_stat_cg),
                            "p_value": float(p_val_cg),
                        },
                        "naive_vs_static": {
                            "mean_delta_ler": mean_delta_naive,
                            "relative_change_pct": float(mean_delta_naive / max(np.mean(static_lers), 1e-9) * 100),
                        },
                    },
                }

    # Compute distance scaling Lambda = LER(d=3) / LER(d=5)
    lambda_scaling: dict[str, Any] = {}
    for sc in scenarios:
        sc_id = sc["id"]
        # Use primary window size W=2000
        w_primary = "w2000"
        d3_cg = aggregates["d3"][sc_id][w_primary]["arms"]["cg_ccads"]["mean_ler"]
        d5_cg = aggregates["d5"][sc_id][w_primary]["arms"]["cg_ccads"]["mean_ler"]
        d3_stat = aggregates["d3"][sc_id][w_primary]["arms"]["static_mwpm"]["mean_ler"]
        d5_stat = aggregates["d5"][sc_id][w_primary]["arms"]["static_mwpm"]["mean_ler"]

        lambda_cg = d3_cg / max(d5_cg, 1e-9)
        lambda_stat = d3_stat / max(d5_stat, 1e-9)

        lambda_scaling[sc_id] = {
            "lambda_static": float(lambda_stat),
            "lambda_cg_ccads": float(lambda_cg),
            "suppression_gain": float(lambda_cg / max(lambda_stat, 1e-9)),
        }

    output_payload = {
        "metadata": {
            "experiment_name": config["experiment"]["name"],
            "version": config["experiment"]["version"],
            "provenance": "simulation",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "execution_duration_sec": round(elapsed_sec, 2),
            "seeds": [seed_range[0], seed_range[1] - 1],
            "total_runs": total_runs,
        },
        "distance_scaling_lambda": lambda_scaling,
        "aggregates": aggregates,
        "trajectories": all_records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    logger.info("Saved complete results to %s", output_path)

    # Manifest
    results_hash = hashlib.sha256(json.dumps(output_payload).encode("utf-8")).hexdigest()
    manifest_payload = {
        "provenance": "simulation",
        "git_commit": get_git_commit_hash(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": " ".join(sys.argv),
        "execution_duration_sec": round(elapsed_sec, 2),
        "system": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "stim_version": stim.__version__,
            "pymatching_version": pymatching.__version__,
        },
        "config": {
            "path": str(config_path),
            "sha256": config_hash,
        },
        "results_file": {
            "path": str(output_path),
            "sha256": results_hash,
            "total_records": len(all_records),
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)
    logger.info("Saved manifest to %s", manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Confirmatory Adaptive QEC Suite")
    parser.add_argument("--config", type=Path, default=Path("configs/suite.yaml"), help="Config file path")
    parser.add_argument("--seeds", type=str, default="1000:1030", help="Seed range in format START:END")
    parser.add_argument("--output", type=Path, default=Path("data/suite_results.json"), help="Output JSON path")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"), help="Manifest JSON path")
    args = parser.parse_args()

    parts = [int(p) for p in args.seeds.split(":")]
    seed_range = (parts[0], parts[1])
    run_suite(args.config, seed_range, args.output, args.manifest)


if __name__ == "__main__":
    main()
