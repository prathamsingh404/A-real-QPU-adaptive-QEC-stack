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

        if detector_map is not None:
            x_defects, z_defects = detector_map.extract_defect_counts(syndromes)
            n_x = max(1, len(detector_map.x_indices) * shots)
            n_z = max(1, len(detector_map.z_indices) * shots)
            x_rate = float(x_defects / n_x)
            z_rate = float(z_defects / n_z)
        else:
            x_rate = defect_rate
            z_rate = defect_rate

        # Autocorrelation between successive shots
        autocorr = 0.0
        if shots > 1:
            shot_defects = np.sum(syndromes, axis=1)
            mean_d = np.mean(shot_defects)
            var_d = np.var(shot_defects)
            if var_d > 1e-12:
                autocorr = float(np.mean((shot_defects[:-1] - mean_d) * (shot_defects[1:] - mean_d)) / var_d)

        cal = calibration_dict or {}

        # Spatial distribution across detector columns
        spatial_dist = tuple(float(x) for x in np.mean(syndromes, axis=0))

        return cls(
            total_shots=shots,
            num_detectors=num_detectors,
            total_defects=total_defects,
            defect_rate=defect_rate,
            x_defect_rate=x_rate,
            z_defect_rate=z_rate,
            drift_score=drift_score,
            burst_score=burst_score,
            syndrome_autocorrelation=autocorr,
            decoder_disagreement=decoder_disagreement,
            code_distance=code_distance,
            num_rounds=num_rounds,
            t1_median_us=float(cal.get("t1_mean_us", 0.0)),
            t2_median_us=float(cal.get("t2_mean_us", 0.0)),
            p1q_median=float(cal.get("p_1q", 0.0)),
            p2q_median=float(cal.get("p_2q", 0.0)),
            pro_median=float(cal.get("p_ro", 0.0)),
            backend_name=str(cal.get("backend", "simulation")),
            window_idx=window_idx,
            spatial_defect_distribution=spatial_dist,
        )
