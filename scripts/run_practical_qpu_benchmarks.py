"""
Execution runner for practical real-world quantum workloads on IBM Heron QPU (ibm_marrakesh).

Runs:
1. Part 1: Molecular Quantum Chemistry via IQPE (H2 ground-state energy).
2. Part 2: Deterministic Quantum Teleportation (state transfer across heavy-hex links).

Zero simulated data. Pure physical QPU execution with verifiable IBM Runtime Job IDs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from adaptive_qec.experiments.molecular_iqpe import MolecularIQPE
from adaptive_qec.experiments.deterministic_teleportation import (
    DeterministicTeleportation,
    get_cardinal_state,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("practical_qpu_runner")


def main() -> None:
    load_dotenv()

    token = os.getenv("IBM_QUANTUM_TOKEN")
    instance = os.getenv("IBM_QUANTUM_INSTANCE")
    channel = os.getenv("IBM_QUANTUM_CHANNEL", "ibm_cloud")
    backend_name = os.getenv("IBM_QUANTUM_BACKEND", "ibm_marrakesh")

    if not token or not instance:
        logger.error("Missing IBM Quantum credentials in .env. Exiting.")
        sys.exit(1)

    logger.info("Initializing QiskitRuntimeService...")
    service = QiskitRuntimeService(channel=channel, token=token, instance=instance)
    backend = service.backend(backend_name)
    logger.info(f"Targeting QPU: {backend.name} ({backend.num_qubits} qubits, Heron r2)")

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    sampler = SamplerV2(mode=backend)

    results_payload: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "backend": backend_name,
        "experiments": {},
    }

    # =========================================================================
    # PART 1: Molecular Quantum Chemistry IQPE (H2 Molecule at Equilibrium)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PART 1: Molecular Quantum Chemistry (H2 Ground State Energy via IQPE)")
    logger.info("=" * 70)
