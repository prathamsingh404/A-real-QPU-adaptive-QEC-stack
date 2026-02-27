"""
Offline Paired Decoder Replay for IBM Hardware Runs (TASK-022, Point 85).

Replays raw hardware execution bitstrings through all baseline ladder decoders
(B0: Factory MWPM, B1: Optimized MWPM, B2: Union-Find, B3: Reweighted MWPM)
under strict paired comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import stim

from adaptive_qec.evaluation.baseline_ladder import BaselineLadder, BaselineRung
from adaptive_qec.evaluation.counterfactual import CounterfactualEngine


def replay_hardware_bitstrings(
    results_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Replay hardware results through baseline ladder decoders."""
    if not results_path.exists():
        raise FileNotFoundError(f"Hardware results not found: {results_path}")
