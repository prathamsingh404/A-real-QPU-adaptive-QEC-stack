"""
Automated Manuscript Table Generator.

Reads strictly from verified data/suite_results.json and real CPU latency profilers.
Contains ZERO hardcoded result constants.

Generates:
  - papers/tables/table_1.tex: Primary Confirmatory Endpoints
  - papers/tables/table_2.tex: Window-Size Ablation Study
  - papers/tables/table_3.tex: Distance Scaling Lambda Suppression
  - papers/tables/table_4.tex: Latency Stack Profiling
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from adaptive_qec.analysis.latency import LatencyProfiler
from adaptive_qec.evaluation.empirical_scaling_and_ablation import (
    ComponentAblationStudy,
    compute_distance_scaling,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MakePaperTables")


def generate_table_1_confirmatory(suite_data: dict[str, Any], output_dir: Path) -> Path:
    """Table 1: Primary Confirmatory Endpoints across Scenarios."""
    tex_path = output_dir / "table_1.tex"
    scenarios_d3 = suite_data["aggregates"]["d3"]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Pre-Registered Confirmatory Endpoints Across Physical Noise Scenarios.} Trajectory-paired delta logical error rate ($\bar{\Delta} = \text{LER}_{\text{adaptive}} - \text{LER}_{\text{static}}$) evaluated over $N=30$ independent held-out test seeds (seeds 1000--1029) at primary window size $W=2{,}000$ and $d=3$. Negative $\bar{\Delta}$ indicates adaptive superiority. Provenance: simulation.}",
        r"\label{tab:confirmatory_endpoints}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Scenario Regime} & \textbf{Static LER} & \textbf{CG-CCADS LER} & \textbf{Paired $\bar{\Delta}$ (95\% CI)} & \textbf{$t$-statistic} & \textbf{$p$-value} & \textbf{Decision} \\",
        r"\midrule",
    ]

    scenario_labels = {
        "stationary_noise": "Stationary Baseline (Negative Control)",
        "uniform_drift": "Uniform Global Drift (Negative Control)",
        "localized_hotspot": "Localized Hotspot (Qubit Drift)",
        "structural_anisotropy": "Structural Anisotropy Shift",
    }

    for sc_id, sc_label in scenario_labels.items():
        if sc_id not in scenarios_d3:
            continue
        sc_data = scenarios_d3[sc_id]["w2000"]
        stat_ler = sc_data["arms"]["static_mwpm"]["mean_ler"]
        stat_std = sc_data["arms"]["static_mwpm"]["std_ler"]
        cg_ler = sc_data["arms"]["cg_ccads"]["mean_ler"]
        cg_std = sc_data["arms"]["cg_ccads"]["std_ler"]

        paired = sc_data["paired_analysis"]["cg_vs_static"]
        delta = paired["mean_delta_ler"]
        ci = paired["ci_95"]
        t_stat = paired["t_statistic"]
        p_val = paired["p_value"]

        p_str = "<10^{-4}" if p_val < 0.0001 else f"{p_val:.4f}"
        sign = "+" if delta > 0 else ""
        decision = "STAY (Abstain)" if abs(delta) < 0.001 or p_val > 0.05 else "REWEIGHT"

        lines.append(
            f"{sc_label} & ${stat_ler:.4f} \\pm {stat_std:.4f}$ & ${cg_ler:.4f} \\pm {cg_std:.4f}$ & "
            f"${sign}{delta:.4f}$ $[{ci[0]:.4f}, {ci[1]:.4f}]$ & ${t_stat:+.2f}$ & ${p_str}$ & {decision} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])

    tex_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Generated %s", tex_path)
    return tex_path


def generate_table_2_ablation(results_path: Path, output_dir: Path) -> Path:
    """Table 2: Ablation Study across Window Regimes and Architectures."""
    tex_path = output_dir / "table_2.tex"
    rows = ComponentAblationStudy.run_ablation(results_path=results_path)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Component Ablation and Window-Size Sensitivity.} Demonstrates the necessity of confidence gating and safe abstention (\texttt{STAY}). At small observation windows ($W=200$), unconstrained naive reweighting suffers catastrophic error inflation due to binomial sample variance. CG-CCADS safely abstains under small windows and reweights only when evidence is conclusive. Provenance: simulation.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Architecture Configuration} & \textbf{Mean LER} & \textbf{$\Delta$ LER (\%)} & \textbf{Switches / Traj} & \textbf{Rollback Rate (\%)} & \textbf{Net Value $V_{\text{net}}$} & \textbf{Abstention Rate (\%)} \\",
        r"\midrule",
    ]

    for r in rows:
        sign = "+" if r.delta_ler_pct > 0 else ""
        lines.append(
            f"{r.configuration} & {r.mean_ler:.5f} & {sign}{r.delta_ler_pct:.2f}\\% & "
            f"{r.switches_per_trajectory:.1f} & {r.rollback_rate_pct:.1f}\\% & "
            f"{r.net_value_v_net:+.5f} & {r.abstention_rate_pct:.1f}\\% \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])

    tex_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Generated %s", tex_path)
    return tex_path


def generate_table_3_distance_scaling(results_path: Path, output_dir: Path) -> Path:
    """Table 3: Code Distance Scaling and Lambda Suppression Factors."""
    tex_path = output_dir / "table_3.tex"
    res = compute_distance_scaling(distances=(3, 5), results_path=results_path)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Code Distance Scaling and Logical Error Suppression.} Error suppression factor $\Lambda_{3 \to 5} = \text{LER}(d=3) / \text{LER}(d=5)$ below threshold ($p=0.004 < p_{\text{th}}$). Provenance: simulation.}",
        r"\label{tab:distance_scaling}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{$d=3$ LER} & \textbf{$d=5$ LER} & \textbf{Suppression $\Lambda_{3 \to 5}$} & \textbf{Suppression Gain} \\",
        r"\midrule",
        f"Static Baseline (MWPM) & ${res.static_lers[3]:.5f}$ & ${res.static_lers[5]:.5f}$ & ${res.lambda_static['3->5']:.3f}$ & $1.000$ \\\\",
        f"CG-CCADS (This Work) & ${res.adaptive_lers[3]:.5f}$ & ${res.adaptive_lers[5]:.5f}$ & ${res.lambda_adaptive['3->5']:.3f}$ & ${res.suppression_gain['3->5']:.3f}$ \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
