"""
Online regret analysis and adaptive efficiency evaluation for QEC controllers.

Computes:
- Online cumulative regret R_T = sum_{t=1}^T [ L(a_t) - L(a^*) ] + sum_{switches} C_switch
- Full cumulative regret curves across multiple evaluation trajectories
- Adaptive efficiency eta = (LER_static - LER_adaptive) / (LER_static - LER_oracle) in [0, 1]
- Publication-quality visualization of empirical regret trajectories
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegretSummary:
    """Summary metrics of online regret."""
    total_regret: float
    time_averaged_regret: float
    switching_cost_total: float
    num_switches: int
    steps: int
    regret_curve: list[float] = field(default_factory=list)


def compute_online_regret(
    arm_lers: Sequence[float],
    oracle_lers: Sequence[float],
    window_size: int = 500,
    adaptation_tax: float = 0.003,
    chosen_actions: Optional[Sequence[Any]] = None,
) -> RegretSummary:
    """Compute proper online regret against dynamic oracle including switching tax.

    R_T = sum_{t=1}^T max(0, L_t(a_t) - L_t(a^*)) + C_switch * N_switches

    Args:
        arm_lers: Logical error rates of the chosen policy arm across windows.
        oracle_lers: Logical error rates of the optimal oracle baseline.
        window_size: Number of shots per syndrome evaluation window.
        adaptation_tax: Switching penalty per policy change.
        chosen_actions: Optional sequence of arm/policy identifiers to detect switches.
                        If None, switches are inferred when consecutive LERs differ.

    Returns:
        RegretSummary with cumulative regret, time-averaged regret, and switch statistics.
    """
    arm_arr = np.asarray(arm_lers, dtype=np.float64)
    oracle_arr = np.asarray(oracle_lers, dtype=np.float64)
    if len(arm_arr) != len(oracle_arr):
        raise ValueError("arm_lers and oracle_lers must have the same length")

    t_steps = len(arm_arr)
    if t_steps == 0:
        return RegretSummary(0.0, 0.0, 0.0, 0, 0, [])

    num_switches = 0
    if chosen_actions is not None and len(chosen_actions) == t_steps:
        for t in range(1, t_steps):
            if chosen_actions[t] != chosen_actions[t - 1]:
                num_switches += 1
    else:
        for t in range(1, t_steps):
            if abs(arm_arr[t] - arm_arr[t - 1]) > 1e-4:
                num_switches += 1

    regret_curve = []
    cum = 0.0
    for t in range(t_steps):
        loss_diff = max(0.0, float(arm_arr[t] - oracle_arr[t]))
        cum += loss_diff
        regret_curve.append(cum)

    total_switching_tax = num_switches * adaptation_tax
    final_total_regret = cum + total_switching_tax
    regret_curve[-1] = final_total_regret

    return RegretSummary(
        total_regret=final_total_regret,
        time_averaged_regret=final_total_regret / t_steps,
        switching_cost_total=total_switching_tax,
        num_switches=num_switches,
        steps=t_steps,
        regret_curve=regret_curve,
    )


def compute_cumulative_regret_curve(
    trajectories: Sequence[Sequence[float]] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute aggregate cumulative regret curve across multiple seeds / runs.

    Args:
        trajectories: 2D array or list of 1D regret sequences (runs x steps).

    Returns:
        Dict with 'mean', 'median', 'std', 'p25', 'p75', 'p95' arrays over time.
    """
    arr = np.asarray(trajectories, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    mean_curve = np.mean(arr, axis=0)
    median_curve = np.median(arr, axis=0)
    std_curve = np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean_curve)
    p25 = np.percentile(arr, 25, axis=0)
    p75 = np.percentile(arr, 75, axis=0)
    p95 = np.percentile(arr, 95, axis=0)

    return {
        "mean": mean_curve,
        "median": median_curve,
        "std": std_curve,
        "p25": p25,
        "p75": p75,
        "p95": p95,
    }


def compute_adaptive_efficiency(
    cg_lers: Sequence[float] | float,
    oracle_lers: Sequence[float] | float,
    static_lers: Sequence[float] | float,
) -> dict[str, float]:
    """Compute adaptive efficiency metric:
        eta = V_gross / V_oracle = (LER_static - LER_cg) / (LER_static - LER_oracle) in [0, 1].

    Args:
        cg_lers: Confidence-gated / adaptive controller LER(s).
        oracle_lers: Hindsight oracle optimal LER(s).
        static_lers: Static unadapted baseline LER(s).

    Returns:
        Dict with 'efficiency', 'gross_value', 'oracle_headroom', 'valid_range'.
    """
    cg_mean = float(np.mean(cg_lers))
    oracle_mean = float(np.mean(oracle_lers))
    static_mean = float(np.mean(static_lers))

    v_gross = static_mean - cg_mean
    v_oracle = static_mean - oracle_mean

    if v_oracle <= 1e-9:
        eta = 1.0 if abs(v_gross) <= 1e-9 else 0.0
    else:
        eta = v_gross / v_oracle

    clamped_eta = float(np.clip(eta, 0.0, 1.0))

    return {
        "efficiency": clamped_eta,
        "raw_efficiency": float(eta),
        "gross_value": v_gross,
        "oracle_headroom": v_oracle,
        "valid_range": (0.0 <= eta <= 1.0),
    }


def plot_regret_curve(
    curve: Sequence[float] | dict[str, np.ndarray],
    output_path: Union[str, Path],
    title: str = "Cumulative Regret R_T vs Time",
) -> Path:
    """Plot cumulative regret trajectory and save to file.

    Args:
        curve: Regret curve values (or dict from compute_cumulative_regret_curve).
        output_path: Destination path (.png or .pdf).
        title: Chart title.

    Returns:
        Path to written image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    if isinstance(curve, dict) and "mean" in curve:
        t = np.arange(len(curve["mean"]))
        ax.plot(t, curve["mean"], color="#1f77b4", lw=2, label="Mean Cumulative Regret")
        if "p25" in curve and "p75" in curve:
            ax.fill_between(t, curve["p25"], curve["p75"], color="#1f77b4", alpha=0.2, label="IQR (25-75%)")
    else:
        c_arr = np.asarray(curve, dtype=np.float64)
        t = np.arange(len(c_arr))
        ax.plot(t, c_arr, color="#1f77b4", lw=2, label="Cumulative Regret R_T")

    ax.set_xlabel("Evaluation Window (t)")
    ax.set_ylabel("Cumulative Regret R_T")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)

    logger.info(f"Saved regret curve plot to {out_file}")
    return out_file
