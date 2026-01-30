"""
Deployable Observation Boundary for Adaptive QEC Controllers.

Strictly encapsulates deployment-observable telemetry:
- Syndrome defect statistics (X vs Z parity defects)
- Spatial and temporal defect distributions
- Syndrome autocorrelation and burst scores
- Decoder disagreement metrics
- Public hardware calibration telemetry (T1, T2, gate error distributions)

Prohibits private latent simulation states (true Pauli error rates, hidden leakage flags,
unobservable logical error rates).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from adaptive_qec.qec.geometry import DetectorMap


@dataclass(frozen=True)
class ObservationFrame:
    """Strictly deployable observation frame for online controller decisions.

    Guarantees no oracle information leakage from simulator to controller.
    """
    # 1. Primary syndrome telemetry (measured on-the-fly)
    total_shots: int
    num_detectors: int
    total_defects: int
    defect_rate: float
    x_defect_rate: float
    z_defect_rate: float

    # 2. Time-series and statistical signatures
    drift_score: float = 0.0
    burst_score: float = 0.0
    syndrome_autocorrelation: float = 0.0
    decoder_disagreement: float = 0.0
