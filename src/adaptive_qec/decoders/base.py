"""
Abstract decoder interface.

All decoders are plug-ins:
    Decoder
    ├── lookup
    ├── MWPM
    ├── Union-Find
    ├── Belief Propagation
    ├── OSD
    ├── CNN
    ├── GNN
    ├── Transformer
    └── Adaptive decoder
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Correction:
    """Decoder output: predicted correction for each observable."""
    observable_corrections: np.ndarray  # shape (num_observables,) or (batch, num_observables)
    confidence: Optional[np.ndarray] = None  # P(correction | S) for each shot
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
