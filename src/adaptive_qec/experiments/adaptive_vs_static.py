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
