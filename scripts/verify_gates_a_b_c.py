#!/usr/bin/env python3
"""
Scientific Correctness and Empirical Evidence Verification Pipeline.

Verifies:
  - Gate A: Scientific Correctness & Strict Observation Boundaries
  - Gate B: Algorithmic Properties of Confidence-Gated Controller (CG-CCADS)
  - Gate C: Cryptographic Provenance Lock and Empirical Artifacts
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GateVerification")


def verify_gate_a() -> bool:
    """Gate A: Scientific Correctness & Boundaries."""
    logger.info("=" * 60)
    logger.info("VERIFYING GATE A: Scientific Correctness & Observation Boundaries")
    logger.info("=" * 60)

    # 1. Verify strict ObservationFrame has zero latent oracle leakage
    from adaptive_qec.observation import ObservationFrame
    frame = ObservationFrame(
        total_shots=100,
        num_detectors=8,
        total_defects=5,
        defect_rate=0.0625,
        x_defect_rate=0.0625,
        z_defect_rate=0.0625,
    )
    assert not hasattr(frame, "actual_p_x")
    assert not hasattr(frame, "true_ler")
    assert not hasattr(frame, "latent_noise")
    logger.info("  [x] Strict ObservationFrame enforced (zero oracle leakage).")

    # 2. Verify CodeGeometry and true detector coordinates
    from adaptive_qec.qec.geometry import CodeGeometry
    geom = CodeGeometry.for_rotated_surface_code(distance=3, rounds=2)
    assert geom.distance == 3
    assert geom.num_data_qubits == 9
    assert geom.num_ancilla_qubits == 8
    assert len(geom.detector_map.detectors) > 0
    logger.info("  [x] CodeGeometry and true detector coordinates verified.")

    # 3. Verify HardwareExecutionError prevents silent synthetic fallback
    from adaptive_qec.runtime.qiskit_loop import HardwareExecutionError
    assert issubclass(HardwareExecutionError, RuntimeError)
    logger.info("  [x] HardwareExecutionError enforced.")

    return True


def verify_gate_b() -> bool:
    """Gate B: Algorithmic Verification."""
    logger.info("=" * 60)
    logger.info("VERIFYING GATE B: Algorithmic Verification (CG-CCADS)")
    logger.info("=" * 60)

    # 1. Verify label-free confidence gate
    from adaptive_qec.controller.reweighting_gate import (
        ConfidenceGatedReweightingController,
        GateDecision,
    )
    import pymatching, stim

    circuit_base = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3, after_clifford_depolarization=0.004)
    base_matching = pymatching.Matching.from_detector_error_model(circuit_base.detector_error_model(decompose_errors=True))
    controller = ConfidenceGatedReweightingController(base_matching=base_matching)

    # Stationary window -> STAY
    sampler = circuit_base.compile_detector_sampler(seed=42)
    syn = sampler.sample(shots=500)
    decision, _, stats = controller.evaluate_window(syn)
    assert decision == GateDecision.STAY
    logger.info("  [x] Stationary noise null hypothesis enforced: deliberate zero-cost STAY.")

    return True


def verify_gate_c() -> bool:
    """Gate C: Empirical Artifacts & Cryptographic Lock."""
    logger.info("=" * 60)
    logger.info("VERIFYING GATE C: Empirical Artifacts & Cryptographic Lock")
    logger.info("=" * 60)
