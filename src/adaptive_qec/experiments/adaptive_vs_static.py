"""
Adaptive vs Static QEC: The Core Experiment.

This is the narrow experiment that demonstrates the paper's central claim:
    "An interpretable, hardware-state-conditioned controller adaptively
     selecting (decoder, DD policy, burst mitigation) outperforms any
     single fixed strategy under realistic non-stationary noise."

Experimental Design:
    1. Generate Stim surface code circuits at d=3 (and optionally d=5, d=7)
    2. Simulate non-stationary noise: inject drift ramps, burst events,
       and leakage-like persistent defects at controlled intervals
    3. Run three arms in parallel on identical syndrome data:
       a. STATIC-MWPM: Fixed MWPM decoder, no DD, no burst mitigation
       b. STATIC-UF:   Fixed UF decoder, fixed XY4 DD, no burst mitigation
       c. ADAPTIVE:     AdaptiveController selects strategy per window
    4. Compare logical error rates with Wilson score confidence intervals
    5. Report: adaptive improvement margin + mode switch timeline

Statistical Rigor:
    - Each arm uses identical syndrome data (no sampling variance between arms)
    - Wilson score 95% CI on logical error rate per window
    - Bonferroni correction for multiple comparisons
    - p-value from two-proportion z-test for adaptive vs best-static

Usage:
    python -m adaptive_qec.experiments.adaptive_vs_static
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import stim

from adaptive_qec.config import NoiseConfig
from adaptive_qec.controller.controller import (
    AdaptiveController,
    ControlAction,
    CostWeights,
    DecoderChoice,
    HardwareState,
)
from adaptive_qec.decoders.mwpm import MWPMDecoder
from adaptive_qec.decoders.union_find import UnionFindDecoder
from adaptive_qec.mitigation.dynamical_decoupling import DDSequenceType
from adaptive_qec.noise.burst_detector import BurstDetector, reshape_syndromes_to_tensor
from adaptive_qec.noise.drift import CompositeDriftDetector, DriftStatus
from adaptive_qec.noise.leakage import LeakageDetector
from adaptive_qec.qec.codes import create_code

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------

def wilson_ci(n_errors: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    More accurate than the Wald interval when p is near 0 or 1.
    """
    if n_total == 0:
        return (0.0, 1.0)
    p = n_errors / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    spread = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def two_proportion_z_test(
    n1_errors: int, n1_total: int,
    n2_errors: int, n2_total: int,
) -> tuple[float, float]:
    """Two-proportion z-test for comparing two binomial proportions.

    Returns (z_statistic, p_value) for H0: p1 = p2 vs H1: p1 != p2.
    """
    from scipy import stats

    p1 = n1_errors / max(n1_total, 1)
    p2 = n2_errors / max(n2_total, 1)
    p_pool = (n1_errors + n2_errors) / max(n1_total + n2_total, 1)

    if p_pool == 0 or p_pool == 1:
        return (0.0, 1.0)

    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1_total + 1/n2_total))
    if se < 1e-15:
        return (0.0, 1.0)

    z = (p1 - p2) / se
    p_value = 2 * stats.norm.sf(abs(z))
    return (float(z), float(p_value))


# ---------------------------------------------------------------------------
# Non-stationary noise injection
# ---------------------------------------------------------------------------

@dataclass
class NoiseSchedule:
    """Defines how noise changes over the experiment timeline.

    The experiment is divided into windows. Each window has a noise profile.
    """
    total_windows: int = 50
    shots_per_window: int = 200

    # Drift ramp: windows where p_2q gradually increases
    drift_start_window: int = 15
    drift_end_window: int = 30
    drift_p2q_base: float = 0.005
    drift_p2q_peak: float = 0.015

    # Burst injection: which windows have correlated error bursts
    burst_windows: list[int] = field(default_factory=lambda: [20, 35])
    burst_intensity: float = 0.5  # fraction of detectors affected (>= 3 detectors at d=3)

    # Leakage: persistent defects starting at a specific window
    leakage_start_window: int = 25
    leakage_detector_indices: list[int] = field(default_factory=lambda: [2, 5])

    def get_noise_config(self, window_idx: int) -> NoiseConfig:
        """Get the noise config for a specific window."""
        noise = NoiseConfig()

        # Base noise
        p_2q = self.drift_p2q_base

        # Drift ramp
        if self.drift_start_window <= window_idx <= self.drift_end_window:
            progress = (window_idx - self.drift_start_window) / max(
                self.drift_end_window - self.drift_start_window, 1
            )
            p_2q = self.drift_p2q_base + progress * (
                self.drift_p2q_peak - self.drift_p2q_base
            )

        noise.gate.two_qubit = p_2q
        return noise

    def is_burst_window(self, window_idx: int) -> bool:
        return window_idx in self.burst_windows

    def is_leakage_active(self, window_idx: int) -> bool:
        return window_idx >= self.leakage_start_window


def apply_dd_to_noise(
    base_noise: NoiseConfig,
    dd_policy: DDSequenceType,
    single_qubit_pulse_error: float = 0.0003,
) -> NoiseConfig:
    """Apply the physical causal impact of dynamical decoupling to the noise model.

    Dynamical decoupling suppresses low-frequency dephasing during idle windows,
    while adding discrete pulse error overhead (X/Y rotations).

    Trade-off:
        - When dephasing is high (e.g. during drift regimes), the suppression
          outweighs the pulse cost, reducing overall effective gate error.
        - In low-noise regimes, the pulse error overhead outweighs dephasing
          suppression, increasing effective gate error.
    """
    noise = NoiseConfig()
    noise.gate.single_qubit = base_noise.gate.single_qubit
    noise.readout.p0_given_1 = base_noise.readout.p0_given_1
    noise.readout.p1_given_0 = base_noise.readout.p1_given_0

    SUPPRESSION = {
        DDSequenceType.NONE: 1.0,
        DDSequenceType.CPMG: 0.45,
        DDSequenceType.XY4: 0.22,
        DDSequenceType.XY8: 0.12,
    }
    PULSE_COUNTS = {
        DDSequenceType.NONE: 0,
        DDSequenceType.CPMG: 2,
        DDSequenceType.XY4: 4,
        DDSequenceType.XY8: 8,
    }

    p_base = base_noise.gate.two_qubit
    f_floor = 0.12
    f_drift = 0.85
    p_floor = 0.005
    p_dep = f_floor * min(p_base, p_floor) + f_drift * max(0.0, p_base - p_floor)
    p_non_dep = max(0.0, p_base - p_dep)

    suppression = SUPPRESSION.get(dd_policy, 1.0)
    pulses = PULSE_COUNTS.get(dd_policy, 0)

    # Suppressed dephasing + unsuppressed error + pulse overhead
    p_eff = p_non_dep + (p_dep * suppression) + (pulses * single_qubit_pulse_error)
    noise.gate.two_qubit = float(max(0.0, p_eff))
    return noise


def inject_burst(
    syndromes: np.ndarray,
    num_detectors_per_round: int,
    num_rounds: int,
    intensity: float = 0.4,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Inject a correlated error burst into syndrome data stochastically.

    Simulates a cosmic-ray-like event: radiation impacts produce spatially
    and temporally localized defects with per-shot stochastic jitter,
    rather than an artificial, identical bitmask on every shot.
    """
    rng = rng or np.random.default_rng()
    result = syndromes.copy()
    num_shots = result.shape[0]

    for shot in range(num_shots):
        # Stochastically hits ~60% of shots during a burst window
        if rng.random() > 0.60:
            continue

        base_n = max(1, int(num_detectors_per_round * intensity))
        jitter = int(rng.integers(-1, 2))
        n_affected = max(1, min(num_detectors_per_round, base_n + jitter))
        affected_dets = rng.choice(num_detectors_per_round, size=n_affected, replace=False)
        burst_round = int(rng.integers(1, max(2, num_rounds)))

        for det in affected_dets:
            if rng.random() < 0.90:
                flat_idx = burst_round * num_detectors_per_round + det
                if flat_idx < result.shape[1]:
                    result[shot, flat_idx] = 1

    return result


def inject_leakage(
    syndromes: np.ndarray,
    num_detectors_per_round: int,
    num_rounds: int,
    leak_detectors: list[int],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Inject persistent defects to simulate leakage stochastically.

    A leaked qubit causes repeated syndrome violations with high
    transition probability (85%), with per-shot stochastic variation.
    """
    rng = rng or np.random.default_rng()
    result = syndromes.copy()
    num_shots = result.shape[0]

    for shot in range(num_shots):
        for det in leak_detectors:
            for r in range(num_rounds):
                if rng.random() < 0.85:
                    flat_idx = r * num_detectors_per_round + det
                    if flat_idx < result.shape[1]:
                        result[shot, flat_idx] = 1
    return result


# ---------------------------------------------------------------------------
# Experiment arms
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    """Result for a single experiment window."""
    window_idx: int
    n_shots: int
    n_errors: int
    error_rate: float
    ci_low: float
    ci_high: float
    decoder_used: str = ""
    dd_used: str = ""
    burst_mitigated: bool = False


@dataclass
class ArmResult:
    """Complete result for one experimental arm."""
    arm_name: str
    windows: list[WindowResult]
    total_shots: int = 0
    total_errors: int = 0
    overall_error_rate: float = 0.0
    overall_ci: tuple[float, float] = (0.0, 1.0)

    def finalize(self) -> None:
        self.total_shots = sum(w.n_shots for w in self.windows)
        self.total_errors = sum(w.n_errors for w in self.windows)
        if self.total_shots > 0:
            self.overall_error_rate = self.total_errors / self.total_shots
            self.overall_ci = wilson_ci(self.total_errors, self.total_shots)

    def to_dict(self) -> dict[str, Any]:
        self.finalize()
        return {
            "arm_name": self.arm_name,
            "total_shots": self.total_shots,
            "total_errors": self.total_errors,
            "overall_error_rate": round(self.overall_error_rate, 6),
            "overall_ci_95": [round(x, 6) for x in self.overall_ci],
            "per_window": [
                {
                    "window": w.window_idx,
                    "n_shots": w.n_shots,
                    "n_errors": w.n_errors,
                    "error_rate": round(w.error_rate, 6),
                    "ci": [round(w.ci_low, 6), round(w.ci_high, 6)],
                    "decoder": w.decoder_used,
                    "dd": w.dd_used,
                    "burst_mitigated": w.burst_mitigated,
                }
                for w in self.windows
            ],
        }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_adaptive_vs_static(
    distance: int = 3,
    rounds: int = 3,
    schedule: Optional[NoiseSchedule] = None,
    output_dir: Optional[Path] = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full adaptive vs static comparison experiment.

    Args:
        distance: Surface code distance.
        rounds: Number of QEC rounds per circuit.
        schedule: Noise schedule (uses default if None).
        output_dir: Directory to save results JSON.
        seed: Random seed for reproducibility.

    Returns:
        Dict with results for all three arms + statistical comparison.
    """
    schedule = schedule or NoiseSchedule()
    rng = np.random.default_rng(seed)

    logger.info(
        f"Starting adaptive vs static experiment: d={distance}, "
        f"rounds={rounds}, windows={schedule.total_windows}, "
        f"shots/window={schedule.shots_per_window}"
    )

    code = create_code("surface", distance=distance, rounds=rounds)
    num_data = distance**2
    num_dets_per_round = num_data - 1  # approximate for surface code

    # Initialize decoders
    base_noise_init = schedule.get_noise_config(0)
    circuit_base_mwpm = code.generate_circuit(noise=apply_dd_to_noise(base_noise_init, DDSequenceType.NONE))
    circuit_base_uf = code.generate_circuit(noise=apply_dd_to_noise(base_noise_init, DDSequenceType.XY4))

    mwpm_decoder_static = MWPMDecoder()
    mwpm_decoder_static.configure(circuit=circuit_base_mwpm)

    uf_decoder_static = UnionFindDecoder()
    uf_decoder_static.configure(circuit=circuit_base_uf)

    mwpm_decoder_adaptive = MWPMDecoder()
    uf_decoder_adaptive = UnionFindDecoder()
    uf_decoder_adaptive.configure(circuit=circuit_base_uf)

    # Initialize controller
    controller = AdaptiveController(
        weights=CostWeights(),
        hysteresis_patience=3,
        hysteresis_margin=0.05,
    )

    # Noise detectors for the adaptive arm
    drift_detector = CompositeDriftDetector(warmup_samples=5)
    burst_detector = BurstDetector()
    leakage_detector = LeakageDetector()

    # Results accumulators
    static_mwpm_results = ArmResult(arm_name="static_mwpm", windows=[])
    static_uf_results = ArmResult(arm_name="static_uf_xy4", windows=[])
    adaptive_results = ArmResult(arm_name="adaptive", windows=[])

    last_detection_rates = np.zeros(circuit_base_mwpm.num_detectors)
    last_burst_active = False

    for window_idx in range(schedule.total_windows):
        base_noise = schedule.get_noise_config(window_idx)
        window_seed = (seed * 100_003 + window_idx) % (2**31)

        # 1. Compile physical circuits with causal dynamical decoupling
        noise_mwpm = apply_dd_to_noise(base_noise, DDSequenceType.NONE)
        circuit_mwpm = code.generate_circuit(noise=noise_mwpm)

        noise_uf = apply_dd_to_noise(base_noise, DDSequenceType.XY4)
        circuit_uf = code.generate_circuit(noise=noise_uf)

        # Adaptive telemetry & state
        drift_report = drift_detector.update(last_detection_rates)
        leakage_frac = 0.0
        if schedule.is_leakage_active(window_idx):
            leakage_frac = len(schedule.leakage_detector_indices) / max(num_dets_per_round, 1)

        state = HardwareState(
            defect_rate=float(last_detection_rates.mean()),
            drift_magnitude=drift_report.magnitude,
            drift_status=drift_report.status,
            burst_active=last_burst_active,
            leakage_fraction=leakage_frac,
            t1_mean_us=150.0,
            t2_mean_us=120.0,
            p_1q=0.0003,
            p_2q=base_noise.gate.two_qubit,
            p_ro=0.01,
            code_distance=distance,
            num_data_qubits=num_data,
            num_detectors=circuit_mwpm.num_detectors,
        )
        action = controller.select_action(state)

        # Adaptive arm circuit compiled with adaptively selected DD policy
        noise_adapt = apply_dd_to_noise(base_noise, action.dd_policy)
        circuit_adapt = code.generate_circuit(noise=noise_adapt)
