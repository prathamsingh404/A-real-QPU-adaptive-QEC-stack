"""
Dynamical Decoupling (DD) integration for idle-time noise suppression.

When a qubit is idle during a QEC round (e.g. waiting for a two-qubit gate
on neighbors to finish, or during SWAP routing on heavy-hex), it accumulates
coherent ZZ crosstalk and non-Markovian low-frequency dephasing.

Indiscriminate DD can hurt: each DD pulse incurs gate error. An optimal
framework applies DD selectively — only when the dephasing noise prevented
exceeds the pulse error penalty.

Supported sequences:
    - CPMG: X - X (order 2, basic dephasing echo)
    - XY4: X - Y - X - Y (order 4, robust against pulse rotation errors)
    - XY8: XY4 + rotated XY4 (order 8, high fidelity)

Sources:
    - IBM Qiskit "Orbit" dynamical decoupling framework
    - Georgia Tech / ETH Zurich ADAPT framework (arXiv 2026)
    - Pokharel et al., "Demonstration of algorithmic quantum speedup for
      an abelian hidden subgroup problem with dynamical decoupling" (2023)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import stim

from adaptive_qec.digital_twin.twin import HardwareDigitalTwin

logger = logging.getLogger(__name__)


class DDSequenceType(str, Enum):
    """Dynamical decoupling pulse sequences."""
    NONE = "none"
    CPMG = "cpmg"  # 2 pulses: X - X
    XY4 = "xy4"    # 4 pulses: X - Y - X - Y
    XY8 = "xy8"    # 8 pulses: X - Y - X - Y - Y - X - Y - X

