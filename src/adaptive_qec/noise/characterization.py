"""
Full noise characterization — combines calibration data with detector statistics.

Builds a comprehensive noise profile of the hardware state at experiment time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from adaptive_qec.noise.statistics import (
    DetectorStatistics,
    SpatialCorrelation,
    TemporalCorrelation,
    compute_detector_statistics,
    compute_spatial_correlation,
    compute_temporal_correlation,
)
from adaptive_qec.qpu.base import CalibrationSnapshot

logger = logging.getLogger(__name__)


@dataclass
class TLSDefectState:
    """State of a Two-Level System (TLS) defect fluctuator."""
    state: int  # 0: off-resonance (low decoherence) or 1: on-resonance (high decoherence)
    current_decoherence_rate: float
    time_in_state: float = 0.0


class TLSFluctuator:
    """
    Two-Level System (TLS) defect switching model (random telegraph noise).

    Models a microscopic fluctuator (e.g., in a Josephson junction dielectric
    or substrate interface) switching stochastically between two discrete decoherence
    rates (gamma_0 and gamma_1) via a continuous-time Markov jump process.

    Governed by:
        Transition 0 -> 1 with switching rate rate_01: P(0 -> 1) = 1 - exp(-rate_01 * dt)
        Transition 1 -> 0 with switching rate rate_10: P(1 -> 0) = 1 - exp(-rate_10 * dt)

    Attributes:
        rate_01: Switching rate from state 0 to 1 (Hz or 1/rounds). Must be >= 0.
        rate_10: Switching rate from state 1 to 0 (Hz or 1/rounds). Must be >= 0.
        gamma_0: Discrete decoherence rate in state 0 (e.g., baseline 1/T1 or 1/T2).
        gamma_1: Discrete decoherence rate in state 1 (elevated rate due to defect resonance).
        dt: Time step duration. Must be > 0.
    """

    def __init__(
        self,
        rate_01: float = 0.05,
        rate_10: float = 0.05,
        gamma_0: float = 1e4,
        gamma_1: float = 5e4,
        dt: float = 1.0,
        initial_state: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        if rate_01 < 0 or rate_10 < 0:
            raise ValueError(f"Switching rates must be non-negative: rate_01={rate_01}, rate_10={rate_10}")
        if gamma_0 < 0 or gamma_1 < 0:
            raise ValueError(f"Decoherence rates must be non-negative: gamma_0={gamma_0}, gamma_1={gamma_1}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if initial_state not in (0, 1):
            raise ValueError(f"initial_state must be 0 or 1, got {initial_state}")

        self.rate_01 = float(rate_01)
        self.rate_10 = float(rate_10)
        self.gamma_0 = float(gamma_0)
        self.gamma_1 = float(gamma_1)
        self.dt = float(dt)
        self._state = int(initial_state)
        self._rng = np.random.default_rng(seed)

    @property
    def state(self) -> int:
        return self._state

    @property
    def current_decoherence_rate(self) -> float:
        return self.gamma_1 if self._state == 1 else self.gamma_0

    @property
    def current_rate(self) -> float:
        """Alias for current_decoherence_rate."""
        return self.current_decoherence_rate

    @property
    def p_state_1_steady(self) -> float:
        """Steady-state occupation probability for state 1."""
        denom = self.rate_01 + self.rate_10
        return self.rate_01 / denom if denom > 0 else 0.5

    @property
    def p_state_0_steady(self) -> float:
        """Steady-state occupation probability for state 0."""
        denom = self.rate_01 + self.rate_10
        return self.rate_10 / denom if denom > 0 else 0.5

    @property
    def correlation_time(self) -> float:
        """Autocorrelation decay time tau_c = 1 / (rate_01 + rate_10)."""
        denom = self.rate_01 + self.rate_10
        return 1.0 / denom if denom > 0 else float("inf")

    @property
    def effective_mean_rate(self) -> float:
        """Ensemble steady-state expected decoherence rate."""
        return self.p_state_0_steady * self.gamma_0 + self.p_state_1_steady * self.gamma_1

    def step(self) -> tuple[int, float]:
        """
        Advance the fluctuator by one time step dt.

        Returns:
            Tuple of (new_state, current_decoherence_rate).
        """
        if self._state == 0:
            p_switch = 1.0 - np.exp(-self.rate_01 * self.dt)
            if self._rng.random() < p_switch:
                self._state = 1
        else:
            p_switch = 1.0 - np.exp(-self.rate_10 * self.dt)
            if self._rng.random() < p_switch:
                self._state = 0

        return self._state, self.current_decoherence_rate

    def generate_trajectory(
        self,
        num_steps: int,
        initial_state: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate a telegraph switching trajectory.

        Args:
            num_steps: Number of discrete steps.
            initial_state: Starting state (0 or 1).

        Returns:
            Tuple of (states, decoherence_rates), each of shape (num_steps,).
        """
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        if initial_state is not None:
            if initial_state not in (0, 1):
                raise ValueError(f"initial_state must be 0 or 1, got {initial_state}")
            self._state = int(initial_state)

        states = np.empty(num_steps, dtype=np.uint8)
        rates = np.empty(num_steps, dtype=np.float64)

        p_01 = 1.0 - np.exp(-self.rate_01 * self.dt)
        p_10 = 1.0 - np.exp(-self.rate_10 * self.dt)

        draws = self._rng.random(num_steps)

        curr = self._state
        states[0] = curr
        rates[0] = self.gamma_1 if curr == 1 else self.gamma_0

        for t in range(1, num_steps):
            if curr == 0:
                if draws[t] < p_01:
                    curr = 1
            else:
                if draws[t] < p_10:
                    curr = 0
            states[t] = curr
            rates[t] = self.gamma_1 if curr == 1 else self.gamma_0

        self._state = curr
        return states, rates

    def generate_telegraph_noise(
        self,
        num_steps: int,
        initial_state: Optional[int] = None,
    ) -> np.ndarray:
        """Convenience method returning decoherence rate trajectory array."""
        _, rates = self.generate_trajectory(num_steps, initial_state)
        return rates

    def reset(self, initial_state: int = 0) -> None:
        """Reset the fluctuator to initial state."""
        if initial_state not in (0, 1):
            raise ValueError(f"initial_state must be 0 or 1, got {initial_state}")
        self._state = initial_state


# Alias for telegraph noise model
TLSTelegraphNoise = TLSFluctuator


class TLSEnsemble:
    """
    Ensemble of multiple independent Two-Level System (TLS) fluctuators.

    Produces 1/f-like telegraph noise when switching rates are log-uniformly
    distributed (standard Dutta-Horn model for 1/f noise in solid-state devices).
    """

    def __init__(
        self,
        fluctuators: list[TLSFluctuator],
        base_rate: float = 0.0,
    ) -> None:
        if not fluctuators:
            raise ValueError("fluctuators list cannot be empty")
        self.fluctuators = fluctuators
        self.base_rate = float(base_rate)

    @classmethod
    def create_dutta_horn_ensemble(
        cls,
        num_fluctuators: int = 10,
        min_rate: float = 1e-4,
        max_rate: float = 1e-1,
        gamma_defect: float = 1e4,
        base_rate: float = 1e4,
        dt: float = 1.0,
        seed: Optional[int] = None,
    ) -> TLSEnsemble:
        """Create a log-uniformly distributed ensemble yielding 1/f telegraph noise."""
        log_rates = np.linspace(np.log10(min_rate), np.log10(max_rate), num_fluctuators)
        rates = 10 ** log_rates
        fluctuators = []
        for i, r in enumerate(rates):
            f_seed = None if seed is None else seed + i
            fluctuators.append(
                TLSFluctuator(
                    rate_01=float(r),
                    rate_10=float(r),
                    gamma_0=0.0,
                    gamma_1=float(gamma_defect / np.sqrt(num_fluctuators)),
                    dt=dt,
                    seed=f_seed,
                )
            )
        return cls(fluctuators=fluctuators, base_rate=base_rate)

    def step(self) -> float:
        total = self.base_rate
        for f in self.fluctuators:
            _, r = f.step()
            total += r
        return total

    def generate_trajectory(self, num_steps: int) -> np.ndarray:
        traj = np.full(num_steps, self.base_rate, dtype=np.float64)
        for f in self.fluctuators:
            traj += f.generate_telegraph_noise(num_steps)
        return traj


@dataclass
class NoiseProfile:
    """
    Complete noise profile combining calibration and experimental data.

    This is the unified noise representation used by the adaptive decoder
    and the noise estimator.

    θ_t = {p_X, p_Y, p_Z, p_readout, p_leakage, ...}
    """
    timestamp: str
    backend: str

    # From calibration
    t1_values: Optional[np.ndarray] = None
    t2_values: Optional[np.ndarray] = None
    readout_errors: Optional[np.ndarray] = None
    gate_errors_1q: Optional[np.ndarray] = None
    gate_errors_2q: Optional[dict[tuple[int, int], float]] = None

    # From detector analysis
    detector_stats: Optional[DetectorStatistics] = None
    temporal_corr: Optional[TemporalCorrelation] = None
    spatial_corr: Optional[SpatialCorrelation] = None

    # Estimated noise parameters
    estimated_physical_error_rate: Optional[float] = None
    estimated_readout_error_rate: Optional[float] = None
    estimated_depolarizing_rate: Optional[float] = None
    correlated_noise_detected: bool = False
    leakage_detected: bool = False
    tls_switching_detected: bool = False

    # Summary
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary."""
        d: dict[str, Any] = {
            "timestamp": self.timestamp,
            "backend": self.backend,
            "estimated_physical_error_rate": self.estimated_physical_error_rate,
            "estimated_readout_error_rate": self.estimated_readout_error_rate,
            "estimated_depolarizing_rate": self.estimated_depolarizing_rate,
            "correlated_noise_detected": self.correlated_noise_detected,
            "leakage_detected": self.leakage_detected,
            "tls_switching_detected": self.tls_switching_detected,
        }

        if self.detector_stats:
            d["detector_stats"] = {
                "mean_detection_rate": self.detector_stats.mean_detection_rate,
                "max_detection_rate": self.detector_stats.max_detection_rate,
                "std_detection_rate": self.detector_stats.std_detection_rate,
                "num_hotspots": len(self.detector_stats.hotspot_detectors),
                "hotspot_detectors": self.detector_stats.hotspot_detectors[:20],
            }

        if self.temporal_corr:
            d["temporal_correlation"] = {
                "mean_lag1_correlation": (
                    float(self.temporal_corr.mean_autocorrelation[0])
                    if len(self.temporal_corr.mean_autocorrelation) > 0
                    else 0.0
                ),
                "num_persistent_detectors": len(self.temporal_corr.persistence_detectors),
            }

        if self.spatial_corr:
            d["spatial_correlation"] = {
                "num_significant_pairs": len(self.spatial_corr.significant_pairs),
            }

        d["summary"] = self.summary
        return d


class NoiseCharacterizer:
    """
    Builds a comprehensive noise profile from calibration and experimental data.

    Combines:
    1. Hardware calibration (T1, T2, readout, gate errors)
    2. Detector statistics (rates, correlations)
    3. Temporal correlation analysis
    4. Spatial correlation analysis
    5. Noise parameter estimation
    """

    def characterize(
        self,
        detection_events: np.ndarray,
        num_rounds: int,
        calibration: Optional[CalibrationSnapshot] = None,
        detector_coordinates: Optional[np.ndarray] = None,
    ) -> NoiseProfile:
        """
        Build a complete noise profile.

        Args:
            detection_events: shape (shots, num_detectors)
            num_rounds: number of QEC rounds
            calibration: optional calibration snapshot
            detector_coordinates: optional detector coordinates

        Returns:
            NoiseProfile with all analyses.
        """
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).isoformat()
        backend = calibration.backend_name if calibration else "unknown"

        logger.info("Starting noise characterization...")

        # 1. Detector statistics
        detector_stats = compute_detector_statistics(detection_events)

        # 2. Temporal correlation
        temporal_corr = compute_temporal_correlation(
            detection_events,
            num_rounds=num_rounds,
            max_lag=min(10, num_rounds),
        )

        # 3. Spatial correlation
        spatial_corr = compute_spatial_correlation(
            detection_events,
            detector_coordinates=detector_coordinates,
        )

        # 4. Build profile
        profile = NoiseProfile(
            timestamp=timestamp,
            backend=backend,
            detector_stats=detector_stats,
            temporal_corr=temporal_corr,
            spatial_corr=spatial_corr,
        )

        # 5. Add calibration data
        if calibration:
            profile.t1_values = calibration.t1_values()
            profile.t2_values = calibration.t2_values()
            profile.readout_errors = calibration.readout_errors()

            # Average readout error
            re = profile.readout_errors
            profile.estimated_readout_error_rate = float(re[re > 0].mean()) if np.any(re > 0) else 0.0

            # Extract 2Q gate errors
            profile.gate_errors_2q = {}
            for gc in calibration.gate_calibrations:
                if len(gc.qubits) == 2 and gc.error is not None:
                    profile.gate_errors_2q[gc.qubits] = gc.error

            # Estimate physical error rate from gate errors
            if profile.gate_errors_2q:
                errors_2q = list(profile.gate_errors_2q.values())
                profile.estimated_physical_error_rate = float(np.mean(errors_2q))
                profile.estimated_depolarizing_rate = profile.estimated_physical_error_rate

        # 6. Detect correlated noise signatures
        if temporal_corr and len(temporal_corr.persistence_detectors) > 0:
            persistence_fraction = (
                len(temporal_corr.persistence_detectors) /
                max(1, temporal_corr.autocorrelation.shape[0])
            )
            if persistence_fraction > 0.1:
                profile.correlated_noise_detected = True
                logger.warning(
                    f"Correlated noise detected: {len(temporal_corr.persistence_detectors)} "
                    f"persistent detectors ({persistence_fraction:.1%})"
                )

        # 7. Detect leakage signatures
        if num_rounds >= 3 and detection_events.ndim == 2:
            try:
                from adaptive_qec.noise.burst_detector import reshape_syndromes_to_tensor
                from adaptive_qec.noise.leakage import LeakageDetector
                det_per_round = detection_events.shape[1] // num_rounds
                if det_per_round > 0:
                    # Average over first few shots to build a stable syndrome trace
                    mean_trace = (detection_events[:min(10, len(detection_events))].mean(axis=0) > 0.3).astype(np.uint8)
                    syndrome_tensor = reshape_syndromes_to_tensor(
                        mean_trace, num_rounds, det_per_round
                    )
                    leakage_detector = LeakageDetector()
                    leakage_analysis = leakage_detector.analyze(syndrome_tensor)
                    if len(leakage_analysis.leaked_qubits) > 0:
                        profile.leakage_detected = True
            except Exception as e:
                logger.debug(f"Leakage check skipped: {e}")

        # 8. Summary
        profile.summary = {
            "mean_detection_rate": detector_stats.mean_detection_rate,
            "num_hotspots": len(detector_stats.hotspot_detectors),
            "correlated_noise": profile.correlated_noise_detected,
            "leakage": profile.leakage_detected,
            "estimated_physical_error": profile.estimated_physical_error_rate,
        }

        logger.info(f"Noise characterization complete: {profile.summary}")
        return profile
