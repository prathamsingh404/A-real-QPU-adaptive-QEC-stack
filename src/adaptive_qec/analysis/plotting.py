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
    results_by_controller: dict[str, list[float]],
    output_path: Optional[Path] = None,
    title: str = "Reward Distribution by Controller",
) -> None:
    """Box plots of per-window reward distributions.

    Parameters
    ----------
    results_by_controller : dict
        Mapping of controller name → list of per-window rewards.
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

    names = list(results_by_controller.keys())
    data = [results_by_controller[n] for n in names]
    colors = [_get_color(n) for n in names]

    bp = ax.boxplot(data, labels=names, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Reward (1 − error rate)")
    ax.set_title(title)
    ax.axhline(y=np.mean([np.mean(d) for d in data]), color="gray",
               linestyle=":", alpha=0.5, label="overall mean")

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        logger.info(f"Saved reward distribution plot to {output_path}")
    plt.close()


def plot_error_rate_trajectory(
    window_results: list[dict[str, Any]],
    controller_names: Optional[list[str]] = None,
    output_path: Optional[Path] = None,
    title: str = "Logical Error Rate Over Time",
) -> None:
    """Plot error rate trajectories for multiple controllers.

    Parameters
    ----------
    window_results : list[dict]
        Raw window results from the experiment harness.
    controller_names : list[str], optional
        If provided, only plot these controllers.
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

    # Group by controller
    by_ctrl: dict[str, list[tuple[int, float]]] = {}
    for r in window_results:
        name = r["controller_name"]
        if controller_names and name not in controller_names:
            continue
        by_ctrl.setdefault(name, []).append(
            (r["window_index"], r["logical_error_rate"])
        )

    for name, points in by_ctrl.items():
        points.sort(key=lambda x: x[0])
        steps = [p[0] for p in points]
        rates = [p[1] for p in points]
        color = _get_color(name)

        # Plot with rolling average
        window = min(10, len(rates) // 4)
        if window > 1:
            smoothed = np.convolve(rates, np.ones(window)/window, mode="valid")
            ax.plot(steps[:len(smoothed)], smoothed, label=name, color=color, alpha=0.9)
            ax.fill_between(
                steps, rates, alpha=0.1, color=color
            )
        else:
            ax.plot(steps, rates, label=name, color=color, alpha=0.9)

    ax.set_xlabel("Window (t)")
    ax.set_ylabel("Logical Error Rate")
    ax.set_title(title)
    ax.legend()
    ax.set_yscale("log")

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        logger.info(f"Saved error rate trajectory to {output_path}")
    plt.close()


def plot_arm_selection_timeline(
    window_results: list[dict[str, Any]],
    controller_name: str,
    output_path: Optional[Path] = None,
    title: str = "Arm Selection Timeline",
) -> None:
    """Visualize which arm was selected at each step.

    Parameters
    ----------
    window_results : list[dict]
        Raw window results.
    controller_name : str
        Which controller to visualize.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot")
        return

    plt.rcParams.update(PLOT_STYLE)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    results = [r for r in window_results if r["controller_name"] == controller_name]
    if not results:
        logger.warning(f"No results for controller: {controller_name}")
        return

    steps = [r["window_index"] for r in results]
    arms = [f"{r['action']['decoder']}:{r['action']['dd_policy']}" for r in results]
    rewards = [r["reward"] for r in results]

    # Map arms to integers for plotting
    unique_arms = sorted(set(arms))
    arm_to_idx = {a: i for i, a in enumerate(unique_arms)}
    arm_indices = [arm_to_idx[a] for a in arms]

    # Top: arm selection
    arm_colors = [_get_color(a.split(":")[0]) for a in arms]
    ax1.scatter(steps, arm_indices, c=arm_colors, s=20, alpha=0.7)
    ax1.set_yticks(range(len(unique_arms)))
    ax1.set_yticklabels(unique_arms)
    ax1.set_ylabel("Selected Arm")
    ax1.set_title(f"{title} — {controller_name}")

    # Bottom: reward
    ax2.plot(steps, rewards, color=_get_color(controller_name), alpha=0.5, linewidth=1)
    window = min(10, len(rewards) // 4)
    if window > 1:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
        ax2.plot(steps[:len(smoothed)], smoothed, color=_get_color(controller_name),
                 linewidth=2, label=f"MA({window})")
    ax2.set_xlabel("Window (t)")
    ax2.set_ylabel("Reward")
    ax2.legend()

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        logger.info(f"Saved arm selection timeline to {output_path}")
    plt.close()


def generate_all_plots(
    experiment_result: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Generate all standard plots for an experiment.

    Returns list of paths to generated plot files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    window_results = experiment_result.get("window_results", [])
    if not window_results:
        logger.warning("No window results to plot")
        return generated

    # 1. Error rate trajectory
    p = output_dir / "error_rate_trajectory.png"
    plot_error_rate_trajectory(window_results, output_path=p)
    generated.append(p)

    # 2. Reward distributions
    by_ctrl: dict[str, list[float]] = {}
    for r in window_results:
        by_ctrl.setdefault(r["controller_name"], []).append(r["reward"])
    p = output_dir / "reward_distributions.png"
    plot_reward_distributions(by_ctrl, output_path=p)
    generated.append(p)

    # 3. Arm selection timelines for each controller
    controllers = set(r["controller_name"] for r in window_results)
    for ctrl in controllers:
        p = output_dir / f"arm_timeline_{ctrl}.png"
        plot_arm_selection_timeline(window_results, ctrl, output_path=p)
        generated.append(p)

    return generated
