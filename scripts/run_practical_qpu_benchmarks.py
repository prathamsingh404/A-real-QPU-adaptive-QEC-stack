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

    iqpe_unmitigated = MolecularIQPE(r_angstrom=0.7414, num_bits=3, tau=1.0, apply_dd=False)
    iqpe_mitigated = MolecularIQPE(r_angstrom=0.7414, num_bits=3, tau=1.0, apply_dd=True)

    qc_iqpe_raw = iqpe_unmitigated.build_circuit()
    qc_iqpe_dd = iqpe_mitigated.build_circuit()

    isa_iqpe_raw = pm.run(qc_iqpe_raw)
    isa_iqpe_dd = pm.run(qc_iqpe_dd)

    logger.info("Submitting Part 1 circuits to ibm_marrakesh...")
    job_iqpe_raw = sampler.run([isa_iqpe_raw], shots=1000)
    job_iqpe_dd = sampler.run([isa_iqpe_dd], shots=1000)

    logger.info(f"  [Part 1A] Raw IQPE Job ID:    {job_iqpe_raw.job_id()}")
    logger.info(f"  [Part 1B] Mitigated IQPE Job ID: {job_iqpe_dd.job_id()}")

    # =========================================================================
    # PART 2: Deterministic Quantum Teleportation (Target State: |+>)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PART 2: Deterministic Quantum Teleportation across Heavy-Hex")
    logger.info("=" * 70)

    target_state = get_cardinal_state("|+>")
    dt_dynamic = DeterministicTeleportation(target_state=target_state, regime="dynamic", apply_dd=True)
    dt_post = DeterministicTeleportation(target_state=target_state, regime="post_selected", apply_dd=False)
    dt_swap = DeterministicTeleportation(target_state=target_state, regime="swap_network", apply_dd=False)

    # Tomography circuits: X, Y, Z basis
    circuits_teleport = [
        dt_dynamic.build_circuit("X"),
        dt_dynamic.build_circuit("Y"),
        dt_dynamic.build_circuit("Z"),
        dt_post.build_circuit("X"),
        dt_post.build_circuit("Y"),
        dt_post.build_circuit("Z"),
        dt_swap.build_circuit("X"),
        dt_swap.build_circuit("Y"),
        dt_swap.build_circuit("Z"),
    ]

    isa_circuits_teleport = pm.run(circuits_teleport)

    logger.info("Submitting Part 2 circuits to ibm_marrakesh...")
    job_teleport = sampler.run(isa_circuits_teleport, shots=1000)
    logger.info(f"  [Part 2] Teleportation Tomography Job ID: {job_teleport.job_id()}")

    # Record job IDs immediately
    jobs_summary = {
        "job_iqpe_raw_id": job_iqpe_raw.job_id(),
        "job_iqpe_dd_id": job_iqpe_dd.job_id(),
        "job_teleport_id": job_teleport.job_id(),
    }
    
    out_dir = Path("data/hardware_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ibm_marrakesh_practical_jobs_submitted.json", "w") as f:
        json.dump(jobs_summary, f, indent=2)

    logger.info("\nAll practical workload jobs successfully submitted to IBM Quantum Runtime!")
    logger.info("Polling for completion and processing physical observables...")

    # Wait for Part 1
    res_iqpe_raw = job_iqpe_raw.result()
    res_iqpe_dd = job_iqpe_dd.result()

    counts_raw = res_iqpe_raw[0].data.c.get_counts()
    counts_dd = res_iqpe_dd[0].data.c.get_counts()

    analysis_raw = iqpe_unmitigated.decode_energy_from_bitstrings(counts_raw)
    analysis_dd = iqpe_mitigated.decode_energy_from_bitstrings(counts_dd)

    logger.info("\n=== PART 1 RESULTS (MOLECULAR GROUND ENERGY) ===")
    logger.info(f"Exact H2 Energy:      {analysis_raw['exact_ground_energy_hartree']:.6f} Hartree")
    logger.info(f"Unmitigated IQPE:    {analysis_raw['qpu_ground_energy_hartree']:.6f} Hartree (Error: {analysis_raw['error_hartree']:.6f})")
    logger.info(f"Adaptive DD Mitigated:{analysis_dd['qpu_ground_energy_hartree']:.6f} Hartree (Error: {analysis_dd['error_hartree']:.6f})")

    # Wait for Part 2
    res_teleport = job_teleport.result()
    fid_dynamic = dt_dynamic.compute_fidelity_from_bitstrings(
        out_x=res_teleport[0].data.c_out.get_bitstrings(),
        bsm_x=res_teleport[0].data.c_bsm.get_bitstrings(),
        out_y=res_teleport[1].data.c_out.get_bitstrings(),
        bsm_y=res_teleport[1].data.c_bsm.get_bitstrings(),
        out_z=res_teleport[2].data.c_out.get_bitstrings(),
        bsm_z=res_teleport[2].data.c_bsm.get_bitstrings(),
    )

    fid_post = dt_post.compute_fidelity_from_bitstrings(
        out_x=res_teleport[3].data.c_out.get_bitstrings(),
        bsm_x=res_teleport[3].data.c_bsm.get_bitstrings(),
        out_y=res_teleport[4].data.c_out.get_bitstrings(),
        bsm_y=res_teleport[4].data.c_bsm.get_bitstrings(),
        out_z=res_teleport[5].data.c_out.get_bitstrings(),
        bsm_z=res_teleport[5].data.c_bsm.get_bitstrings(),
    )

    fid_swap = dt_swap.compute_fidelity_from_bitstrings(
        out_x=res_teleport[6].data.c_out.get_bitstrings(),
        bsm_x=None,
        out_y=res_teleport[7].data.c_out.get_bitstrings(),
        bsm_y=None,
        out_z=res_teleport[8].data.c_out.get_bitstrings(),
        bsm_z=None,
    )

    logger.info("\n=== PART 2 RESULTS (QUANTUM TELEPORTATION FIDELITY) ===")
    logger.info(f"Classical Entanglement Bound: {fid_dynamic['classical_bound']:.4f}")
    logger.info(f"Dynamic Teleportation Fidelity: {fid_dynamic['state_fidelity']:.4f} (Yield: {fid_dynamic['deterministic_yield_rate']*100:.1f}%)")
    logger.info(f"Post-Selected Fidelity:        {fid_post['state_fidelity']:.4f} (Yield: {fid_post['deterministic_yield_rate']*100:.1f}%)")
    logger.info(f"SWAP Network Fidelity:         {fid_swap['state_fidelity']:.4f} (Yield: {fid_swap['deterministic_yield_rate']*100:.1f}%)")

    results_payload["experiments"]["part1_molecular_iqpe"] = {
        "job_id_raw": job_iqpe_raw.job_id(),
        "job_id_dd": job_iqpe_dd.job_id(),
        "raw_results": analysis_raw,
        "mitigated_results": analysis_dd,
    }
    results_payload["experiments"]["part2_teleportation"] = {
        "job_id": job_teleport.job_id(),
        "dynamic_feedforward": fid_dynamic,
        "post_selected": fid_post,
        "swap_network": fid_swap,
    }

    final_results_path = out_dir / "ibm_marrakesh_practical_benchmarks_results.json"
    with open(final_results_path, "w") as f:
        json.dump(results_payload, f, indent=2)

    logger.info(f"\nAll empirical results saved to: {final_results_path}")


if __name__ == "__main__":
    main()
