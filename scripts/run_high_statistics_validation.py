"""
High-Statistics Power Test: 50,000 shots per arm on distance-3 surface code.
Establishes statistical power and p-value for adaptive vs static QEC under non-stationary noise.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time

from adaptive_qec.experiments.adaptive_vs_static import (
    NoiseSchedule,
    run_adaptive_vs_static,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("high_statistics_validation")

def main():
    logger.info("Initializing 50,000-shot high-statistics validation experiment...")
    # 50 windows x 1000 shots = 50,000 shots per arm
    schedule = NoiseSchedule(total_windows=50, shots_per_window=1000)
    
    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    start_t = time.time()
    results = run_adaptive_vs_static(
        distance=3,
        rounds=3,
        schedule=schedule,
        output_dir=out_dir,
        seed=42,
    )
    elapsed = time.time() - start_t
    logger.info(f"Experiment completed in {elapsed:.2f}s")
    
    # Save a standardized named file for reporting
    standard_out = out_dir / "adaptive_vs_static_high_stats_50k.json"
    with open(standard_out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"High-statistics benchmark saved to {standard_out}")

if __name__ == "__main__":
    main()
