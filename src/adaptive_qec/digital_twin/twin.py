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

