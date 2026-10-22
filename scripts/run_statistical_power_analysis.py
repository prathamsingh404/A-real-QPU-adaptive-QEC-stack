#!/usr/bin/env python3
"""
Pre-Registered Statistical Power Analysis Script.
Task: TASK-082 [Point 93]
Ensures >= 80% statistical power at alpha = 0.05 for confirmatory trajectory-paired comparisons.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from adaptive_qec.analysis.statistical_rigor import (
    ConfirmatoryProtocol,
    PowerAnalysis,
    SeedPartition,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PowerAnalysis")


def run_power_analysis() -> dict:
    logger.info("Running Pre-Registered Confirmatory Statistical Power Analysis...")

    # 1. Verify seed partition integrity
    SeedPartition.validate_no_overlap()
    train_seeds = SeedPartition.get_seeds("train")
    val_seeds = SeedPartition.get_seeds("validation")
    test_seeds = SeedPartition.get_seeds("test")

    logger.info("Seed Partition Validated:")
    logger.info("  Train seeds: %s (N=%d)", train_seeds, len(train_seeds))
    logger.info("  Validation seeds: %s (N=%d)", val_seeds, len(val_seeds))
    logger.info("  Test seeds: %s (N=%d)", test_seeds, len(test_seeds))

    n_test = len(test_seeds)  # N = 30
    alpha = 0.05

    # 2. Compute power across a spectrum of effect sizes
    effect_sizes = [0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 1.0, 1.2]
    power_curve = []

    for d in effect_sizes:
        p_two_sided = PowerAnalysis.compute_power(d, n_test, alpha=alpha, alternative="two-sided")
        p_one_sided = PowerAnalysis.compute_power(d, n_test, alpha=alpha, alternative="one-sided")
        n_req_80 = PowerAnalysis.sample_size_for_power(d, target_power=0.80, alpha=alpha)
        n_req_90 = PowerAnalysis.sample_size_for_power(d, target_power=0.90, alpha=alpha)

        power_curve.append({
            "cohens_d": d,
            "power_two_sided_n30": round(p_two_sided, 4),
            "power_one_sided_n30": round(p_one_sided, 4),
            "n_required_for_80_pct_power": n_req_80,
            "n_required_for_90_pct_power": n_req_90,
        })
        logger.info(
            "Effect size d=%.2f -> Power (N=30, alpha=0.05): %.1f%% (two-sided), %.1f%% (one-sided) | Min N for 80%%: %d",
            d,
            p_two_sided * 100,
            p_one_sided * 100,
            n_req_80,
        )

    # 3. Target effect size analysis for QEC delta LER
    # For a scenario with sigma_delta = 0.015 and true benefit delta = -0.010, d = 0.667.
    target_d = 0.60
    target_power = PowerAnalysis.compute_power(target_d, n_test, alpha=alpha)
    meets_80_pct = target_power >= 0.80

    logger.info("Target effect size d=%.2f yields power=%.2f%% (>=80%%: %s)", target_d, target_power * 100, meets_80_pct)

    results = {
        "status": "PASSED" if meets_80_pct else "FAILED",
        "primary_endpoint": ConfirmatoryProtocol.PRIMARY_ENDPOINT,
        "confirmatory_regimes": ConfirmatoryProtocol.CONFIRMATORY_REGIMES,
        "exploratory_regimes": ConfirmatoryProtocol.EXPLORATORY_REGIMES,
        "seed_partition": {
            "train": train_seeds,
            "validation": val_seeds,
            "test": test_seeds,
        },
        "target_effect_size_d": target_d,
        "target_power_n30": round(target_power, 4),
        "meets_80_pct_threshold": meets_80_pct,
        "power_curve": power_curve,
    }

    out_path = Path("data/power_analysis_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved power analysis artifact to %s", out_path)
    return results


if __name__ == "__main__":
    run_power_analysis()
