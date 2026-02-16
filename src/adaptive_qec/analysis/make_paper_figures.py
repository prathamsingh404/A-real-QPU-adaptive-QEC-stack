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
