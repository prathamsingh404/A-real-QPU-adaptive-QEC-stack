"""
Generate Cryptographic SHA-256 Provenance Lock (paper.lock).

Computes real SHA-256 hashes for all empirical simulation data files,
generated manuscript tables, and figures, binding them to the git commit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MakeLock")

TRACKED_FILES = [
    Path("configs/suite.yaml"),
    Path("data/suite_results.json"),
    Path("data/manifest.json"),
    Path("papers/tables/table_1.tex"),
    Path("papers/tables/table_2.tex"),
    Path("papers/tables/table_3.tex"),
    Path("papers/tables/table_4.tex"),
    Path("papers/figures/fig_1_window_variance_tradeoff.pdf"),
    Path("papers/figures/fig_2_architecture_diagram.pdf"),
    Path("papers/figures/fig_4_distance_scaling.pdf"),
    Path("papers/figures/fig_5_paired_deltas.pdf"),
]
