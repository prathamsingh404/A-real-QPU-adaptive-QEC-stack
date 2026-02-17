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
