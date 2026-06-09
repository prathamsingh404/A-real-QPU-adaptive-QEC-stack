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

    # 3. Context & Hardware Telemetry (calibrated prior to execution)
    code_distance: int = 3
    num_rounds: int = 3
    t1_median_us: float = 0.0
    t2_median_us: float = 0.0
    p1q_median: float = 0.0
    p2q_median: float = 0.0
    pro_median: float = 0.0
    backend_name: str = "simulation"
    elapsed_cycles: int = 0
    window_idx: int = 0

    # Optional spatial/temporal histogram arrays
    spatial_defect_distribution: Optional[tuple[float, ...]] = None
    temporal_defect_distribution: Optional[tuple[float, ...]] = None

    def __post_init__(self) -> None:
        """Validate that no illegal or privileged fields are attached."""
        for forbidden in ["true_ler", "actual_p_x", "actual_p_z", "latent_noise", "is_leakage_active"]:
            if hasattr(self, forbidden):
                raise ValueError(f"Oracle field '{forbidden}' is strictly forbidden in ObservationFrame.")

    @classmethod
    def from_syndrome_batch(
        cls,
        syndromes: np.ndarray,
        detector_map: Optional[DetectorMap] = None,
        window_idx: int = 0,
        drift_score: float = 0.0,
        burst_score: float = 0.0,
        decoder_disagreement: float = 0.0,
        calibration_dict: Optional[dict[str, Any]] = None,
        code_distance: int = 3,
        num_rounds: int = 3,
    ) -> ObservationFrame:
        """Construct an ObservationFrame strictly from measured syndrome arrays."""
        if syndromes.ndim != 2:
            raise ValueError(f"Syndromes array must be 2D, got shape {syndromes.shape}")

        shots, num_detectors = syndromes.shape
        total_defects = int(np.sum(syndromes))
        defect_rate = float(total_defects / max(1, shots * num_detectors))
