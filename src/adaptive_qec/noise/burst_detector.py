"""
Correlated error burst detection for QEC syndromes.

Detects spatiotemporal clusters of errors that violate the independent-error
assumption underlying surface codes. Caused by:
    - Cosmic ray impacts (wide spatial, sharp temporal)
    - Quasiparticle poisoning (localized, lingering)
    - Crosstalk events (patterned, gate-correlated)

These bursts are rare (~1/hour) but catastrophic: they can cause correlated
logical failures that defeat error correction.

Detection approach:
    For each time window of width w, compute whether the observed defect
    rate exceeds the null hypothesis (independent Bernoulli at rate p_base)
    using a chi-squared-like test at significance level α.

Sources:
    - Google Quantum AI, "Quantum error correction below the surface code
      threshold" (2025) — cosmic ray discussion
    - McEwen et al., "Resolving catastrophic error bursts from cosmic rays
      in large arrays of superconducting qubits" (2022)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class BurstType(str, Enum):
    """Classification of detected error bursts."""
    COSMIC_RAY = "cosmic_ray"       # Wide spatial, sharp temporal
    QP_POISONING = "qp_poisoning"   # Localized, lingering
    CROSSTALK = "crosstalk"         # Patterned, gate-correlated
    UNKNOWN = "unknown"


@dataclass
class BurstEvent:
    """A detected error burst."""
    round_start: int           # First round of the burst
    round_end: int             # Last round of the burst
