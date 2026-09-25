"""
Publication-quality plotting for adaptive QEC experiments.

Generates figures suitable for research papers:
    1. Regret curves (cumulative regret vs. time)
    2. Controller comparison (reward distributions)
    3. Arm selection heatmaps
    4. Error rate trajectories with noise phase annotations
    5. Statistical significance forest plots

All plots use matplotlib with a consistent, clean style.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Style configuration
PLOT_STYLE = {
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "lines.linewidth": 2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
}

# Color palette
COLORS = {
    "static": "#636363",      # gray
    "cost_based": "#3182bd",  # blue
    "exp3": "#e6550d",        # orange
    "exp3p": "#31a354",       # green
    "da_se": "#756bb1",       # purple
    "sprt": "#de2d26",        # red
    "oracle": "#000000",      # black
}


def _get_color(name: str) -> str:
    """Get color for a controller, with fallback."""
    for key, color in COLORS.items():
        if key in name.lower():
            return color
    # Fallback: hash-based color
    h = hash(name) % 360
    return f"hsl({h}, 70%, 50%)"


def plot_regret_curves(
    regret_analyses: dict[str, Any],
    output_path: Optional[Path] = None,
    title: str = "Cumulative Regret vs. Time",
) -> None:
    """Plot cumulative regret curves for all controllers.

    Parameters
    ----------
    regret_analyses : dict
        Mapping of controller name → RegretAnalysis (or dict with regret_curve).
    output_path : Path, optional
        If provided, save the figure here.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot")
        return

    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots()

    for name, analysis in regret_analyses.items():
        curve = analysis.regret_curve if hasattr(analysis, "regret_curve") else analysis.get("regret_curve", [])
        if not curve:
            continue
        color = _get_color(name)
        ax.plot(curve, label=name, color=color, alpha=0.9)

    ax.set_xlabel("Window (t)")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        logger.info(f"Saved regret plot to {output_path}")
    plt.close()


def plot_reward_distributions(
