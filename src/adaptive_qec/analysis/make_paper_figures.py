"""
Automated Manuscript Figure Generator.

Reads strictly from verified data/suite_results.json.
Contains ZERO hardcoded result constants.

Generates:
  - papers/figures/fig_1_window_variance_tradeoff.pdf/.png
  - papers/figures/fig_2_architecture_diagram.pdf/.png
  - papers/figures/fig_4_distance_scaling.pdf/.png
  - papers/figures/fig_5_paired_deltas.pdf/.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # Headless
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MakePaperFigures")

# Publication style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


def fig_1_window_variance_tradeoff(data: dict[str, Any], out_dir: Path) -> None:
    """Figure 1: Window-Size Estimation Variance vs Drift Benefit."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    d3_data = data["aggregates"]["d3"]
    w_labels = ["W=200", "W=1000", "W=2000"]
    w_keys = ["w200", "w1000", "w2000"]

    # Panel A: Stationary Noise (Negative Control)
    stat_sc = d3_data["stationary_noise"]
    stat_mwpm = [stat_sc[w]["arms"]["static_mwpm"]["mean_ler"] for w in w_keys]
    naive_reweighted = [stat_sc[w]["arms"]["naive_reweighted_mwpm"]["mean_ler"] for w in w_keys]
    cg_ccads = [stat_sc[w]["arms"]["cg_ccads"]["mean_ler"] for w in w_keys]

    x = np.arange(len(w_labels))
    width = 0.26

    ax1.bar(x - width, stat_mwpm, width, label="Static MWPM (Factory)", color="#4a7bb0", edgecolor="black")
    ax1.bar(x, naive_reweighted, width, label="Naive Reweighted (Spitz et al.)", color="#d95f02", edgecolor="black")
    ax1.bar(x + width, cg_ccads, width, label="CG-CCADS (Gated Policy)", color="#2ca02c", edgecolor="black")

    ax1.set_xticks(x)
    ax1.set_xticklabels(w_labels)
    ax1.set_ylabel("Logical Error Rate (LER)")
    ax1.set_title("(a) Stationary Noise: Variance Explosion at Small W")
    ax1.legend(loc="upper right")

    # Panel B: Localized Hotspot Drift
    hot_sc = d3_data["localized_hotspot"]
    hot_stat = [hot_sc[w]["arms"]["static_mwpm"]["mean_ler"] for w in w_keys]
    hot_naive = [hot_sc[w]["arms"]["naive_reweighted_mwpm"]["mean_ler"] for w in w_keys]
    hot_cg = [hot_sc[w]["arms"]["cg_ccads"]["mean_ler"] for w in w_keys]
    hot_oracle = [hot_sc[w]["arms"]["oracle_mwpm"]["mean_ler"] for w in w_keys]

    ax2.plot(w_labels, hot_stat, "o--", label="Static MWPM (Stale Weights)", color="#4a7bb0")
    ax2.plot(w_labels, hot_naive, "s-", label="Naive Reweighted", color="#d95f02")
    ax2.plot(w_labels, hot_cg, "^-", label="CG-CCADS (This Work)", color="#2ca02c", linewidth=2.5)
    ax2.plot(w_labels, hot_oracle, "d:", label="Oracle Matched Bound", color="#7570b3")

    ax2.set_xlabel("Syndrome Observation Window Size (W)")
    ax2.set_ylabel("Logical Error Rate (LER)")
    ax2.set_title("(b) Localized Hotspot: Reweighting Advantage")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_dir / "fig_1_window_variance_tradeoff.png", dpi=300)
    fig.savefig(out_dir / "fig_1_window_variance_tradeoff.pdf")
    plt.close(fig)
    logger.info("Saved fig_1_window_variance_tradeoff")


def fig_2_architecture_diagram(out_dir: Path) -> None:
    """Figure 2: End-to-End System Architecture (Block Schematic)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")

    blocks = [
        ("Quantum Device\n& Syndrome Stream\n$S_t \\in \\{0, 1\\}^{W \\times M}$", (0.05, 0.4), (0.22, 0.35), "#e0f3f8"),
        ("Label-Free Diagnostics\n• CUSUM Drift Score $Z_t$\n• Spitz Edge Shift $\\hat{p}_e$\n• Binomial SE $\\sigma_e$", (0.33, 0.35), (0.28, 0.45), "#fee0d2"),
        ("CG-CCADS 3-Gate Rule\n1. Stationarity: $Z_t \\leq T_{drift}$?\n2. Noise Floor: LCB $\\leq \\epsilon$?\n3. Cost Barrier: LCB $> \\tau_{adapt}$?", (0.65, 0.35), (0.32, 0.45), "#e5f5e0"),
    ]

    for title, (x, y), (w, h), bg in blocks:
        rect = plt.Rectangle((x, y), w, h, facecolor=bg, edgecolor="#252525", linewidth=1.5, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", transform=ax.transAxes, fontweight="bold")

    # Connect arrows
    ax.annotate("", xy=(0.33, 0.57), xytext=(0.27, 0.57), arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=6))
    ax.annotate("", xy=(0.65, 0.57), xytext=(0.61, 0.57), arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=6))

    ax.text(0.5, 0.12, "Strictly Label-Free Pipeline: Zero Peek at Ground-Truth Logical Observables", ha="center", va="center", transform=ax.transAxes, fontstyle="italic", fontsize=11)

    plt.tight_layout()
    fig.savefig(out_dir / "fig_2_architecture_diagram.png", dpi=300)
    fig.savefig(out_dir / "fig_2_architecture_diagram.pdf")
    plt.close(fig)
    logger.info("Saved fig_2_architecture_diagram")


def fig_4_distance_scaling(data: dict[str, Any], out_dir: Path) -> None:
    """Figure 4: Code Distance Scaling (d=3 vs d=5)."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    distances = [3, 5]
    stat_lers = [
        data["aggregates"]["d3"]["stationary_noise"]["w2000"]["arms"]["static_mwpm"]["mean_ler"],
        data["aggregates"]["d5"]["stationary_noise"]["w2000"]["arms"]["static_mwpm"]["mean_ler"],
    ]
    cg_lers = [
        data["aggregates"]["d3"]["stationary_noise"]["w2000"]["arms"]["cg_ccads"]["mean_ler"],
        data["aggregates"]["d5"]["stationary_noise"]["w2000"]["arms"]["cg_ccads"]["mean_ler"],
    ]

    ax.plot(distances, stat_lers, "o--", label="Static MWPM", color="#4a7bb0", markersize=8)
    ax.plot(distances, cg_lers, "s-", label="CG-CCADS", color="#2ca02c", markersize=8)

    ax.set_xticks([3, 5])
    ax.set_xlabel("Code Distance ($d$)")
    ax.set_ylabel("Logical Error Rate (LER)")
    ax.set_yscale("log")
    ax.set_title("Code Distance Scaling Below Threshold ($p=0.004$)")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_dir / "fig_4_distance_scaling.png", dpi=300)
    fig.savefig(out_dir / "fig_4_distance_scaling.pdf")
    plt.close(fig)
    logger.info("Saved fig_4_distance_scaling")


def fig_5_paired_deltas(data: dict[str, Any], out_dir: Path) -> None:
    """Figure 5: Forest Plot of Paired Delta LER across Scenarios from Suite Results."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    scenarios_d3 = data["aggregates"]["d3"]
    scenario_order = ["stationary_noise", "uniform_drift", "localized_hotspot", "structural_anisotropy"]
    scenario_names = {
        "stationary_noise": "Stationary Baseline",
        "uniform_drift": "Uniform Global Drift",
        "localized_hotspot": "Localized Hotspot",
        "structural_anisotropy": "Structural Anisotropy",
    }

    labels = []
    deltas = []
    ci_lows = []
    ci_highs = []

    for sc in scenario_order:
        if sc not in scenarios_d3:
            continue
        paired = scenarios_d3[sc]["w2000"]["paired_analysis"]["cg_vs_static"]
        labels.append(scenario_names[sc])
        deltas.append(paired["mean_delta_ler"])
        ci_lows.append(paired["ci_95"][0])
        ci_highs.append(paired["ci_95"][1])

    y_pos = np.arange(len(labels))
    deltas = np.array(deltas)
    ci_lows = np.array(ci_lows)
    ci_highs = np.array(ci_highs)

    for i in range(len(labels)):
        col = "#2ca02c" if deltas[i] < -0.001 else "#4a7bb0"
        err_low = deltas[i] - ci_lows[i]
        err_high = ci_highs[i] - deltas[i]
        ax.errorbar(
            deltas[i],
            y_pos[i],
            xerr=[[err_low], [err_high]],
            fmt="o",
            color="black",
            ecolor=col,
            elinewidth=2.5,
            capsize=4,
            markersize=6,
        )

    ax.axvline(0.0, color="red", linestyle="--", linewidth=1.5, label=r"Null Effect ($\Delta=0$)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Paired Delta LER: $\Delta = \text{LER}_{\text{adaptive}} - \text{LER}_{\text{static}}$ (95% CI)")
    ax.set_title("Confirmatory Paired Delta LER ($N=30$ Held-Out Seeds, $W=2{,}000$)")
    ax.legend(loc="lower left")

    plt.tight_layout()
    fig.savefig(out_dir / "fig_5_paired_deltas.png", dpi=300)
    fig.savefig(out_dir / "fig_5_paired_deltas.pdf")
    plt.close(fig)
    logger.info("Saved fig_5_paired_deltas")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Paper Figures from Suite Results")
    parser.add_argument("--results", type=Path, default=Path("data/suite_results.json"), help="Suite results JSON path")
    parser.add_argument("--output-dir", type=Path, default=Path("papers/figures"), help="Output directory")
    args = parser.parse_args()

    if not args.results.exists():
        raise FileNotFoundError(f"Missing results file {args.results}. Run experiments/run_suite.py first.")

    with open(args.results, "r", encoding="utf-8") as f:
        data = json.load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_1_window_variance_tradeoff(data, args.output_dir)
    fig_2_architecture_diagram(args.output_dir)
    fig_4_distance_scaling(data, args.output_dir)
    fig_5_paired_deltas(data, args.output_dir)
    logger.info("All paper figures generated successfully.")


if __name__ == "__main__":
    main()
