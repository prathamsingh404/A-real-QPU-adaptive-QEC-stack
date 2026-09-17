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
