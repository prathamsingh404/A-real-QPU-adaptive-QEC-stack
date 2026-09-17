"""
Hardware digital twin.

Internal representation of the QPU:

    Qubit
     ├── T1
     ├── T2
     ├── readout error
     ├── 1Q fidelity
     ├── 2Q fidelity
     ├── leakage probability
     └── temporal behavior

Plus topology:
    q0 ─ q1 ─ q2
         │
         q3

The model predicts:
    P(logical failure) from the current estimated hardware state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from adaptive_qec.qpu.base import CalibrationSnapshot

logger = logging.getLogger(__name__)


@dataclass
class QubitState:
    """Current estimated state of a single qubit."""
    index: int
    t1_us: float = 0.0
    t2_us: float = 0.0
    readout_error: float = 0.0
    single_qubit_fidelity: float = 1.0
    two_qubit_fidelities: dict[int, float] = field(default_factory=dict)  # neighbor → fidelity
    leakage_probability: float = 0.0
    last_updated: str = ""

    # Temporal history
    t1_history: list[tuple[str, float]] = field(default_factory=list)
    t2_history: list[tuple[str, float]] = field(default_factory=list)
    readout_history: list[tuple[str, float]] = field(default_factory=list)


class HardwareDigitalTwin:
    """
    Maintains an internal model of the QPU hardware state.

    Updated from calibration snapshots and experimental observations.
