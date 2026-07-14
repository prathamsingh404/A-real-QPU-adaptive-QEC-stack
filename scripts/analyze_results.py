"""
Post-hoc analysis of saved experiments.

Usage:
    python scripts/analyze_results.py --experiment <experiment_id>
    python scripts/analyze_results.py --compare <id1>,<id2>
    python scripts/analyze_results.py --list
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_qec.data.store import ExperimentStore
from adaptive_qec.analysis.statistics import compare_error_rates


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze AdaptiveQEC experiment results")
    parser.add_argument("--base-path", default="experiments", help="Experiment store path")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--experiment", type=str, help="Show details for an experiment")
    parser.add_argument("--compare", type=str, help="Compare two experiments: id1,id2")

    args = parser.parse_args()
    setup_logging()

    store = ExperimentStore(args.base_path)

    if args.list:
        experiments = store.list_experiments()
        print(f"\nFound {len(experiments)} experiments:")
        for exp_id in experiments:
            try:
                metrics = store.load_metrics(exp_id)
                ler = metrics.get("logical_error_rate", "?")
                print(f"  {exp_id}: LER={ler}")
            except Exception:
                print(f"  {exp_id}: (metrics unavailable)")

    elif args.experiment:
        try:
            metrics = store.load_metrics(args.experiment)
            print(f"\n=== Experiment: {args.experiment} ===")
            print(json.dumps(metrics, indent=2, default=str))
        except FileNotFoundError:
            print(f"Experiment not found: {args.experiment}")
