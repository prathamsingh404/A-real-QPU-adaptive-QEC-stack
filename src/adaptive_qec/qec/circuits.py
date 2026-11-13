"""
Circuit generation and conversion utilities.

Handles:
- Stim → Qiskit circuit conversion for QPU execution
- Detector/observable annotation preservation
- Noise model injection into circuits
- Qubit mapping for hardware topologies
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import stim

from adaptive_qec.config import NoiseConfig, QECConfig
from adaptive_qec.qec.codes import QECCode, create_code

logger = logging.getLogger(__name__)


def build_repetition_code_memory(
    distance: int,
    rounds: int,
    p_error: float = 0.0,
    basis: str = "Z",
) -> stim.Circuit:
    """
    Build a repetition code memory circuit in Stim with proper detector and observable annotations.

    Args:
        distance: Code distance (number of data qubits, odd integer >= 3).
        rounds: Number of syndrome extraction rounds (>= 1).
        p_error: Uniform physical error probability (Pauli / depolarizing).
        basis: Logical memory basis ('Z' for bit-flip protection, 'X' for phase-flip).

    Returns:
        stim.Circuit annotated with detectors and logical observables.
    """
    if distance < 3:
        raise ValueError(f"Distance must be >= 3, got {distance}")
    if rounds < 1:
        raise ValueError(f"Rounds must be >= 1, got {rounds}")

    task = "repetition_code:memory"
    if p_error > 0.0:
        circuit = stim.Circuit.generated(
            task,
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=p_error,
            before_round_data_depolarization=p_error,
            before_measure_flip_probability=p_error,
            after_reset_flip_probability=p_error,
        )
    else:
        circuit = stim.Circuit.generated(
            task,
            distance=distance,
            rounds=rounds,
        )
    return circuit


def convert_stabilizer_schedule_to_cz(
    circuit: stim.Circuit,
    pipelined: bool = True,
) -> stim.Circuit:
    """
    Convert CX-based stabilizer entangling gates to native CZ gates with basis changes.

    On superconducting architectures (e.g. IBM Heron), CZ is often the native 2-qubit gate.
    CX(c, t) is transformed into:
        H(t) -> TICK -> CZ(c, t) -> TICK -> H(t)

    Args:
        circuit: Input Stim circuit with CX gates.
        pipelined: If True, inserts TICK barriers between basis change layers and CZ layers
            to pipeline concurrent multi-qubit interactions across the lattice.

    Returns:
        Converted stim.Circuit with CZ entangling gates.
    """
    new_circuit = stim.Circuit()
    for instruction in circuit.flattened():
        if instruction.name == "CX":
            targets = instruction.targets_copy()
            # targets list has [control0, target0, control1, target1, ...]
            targs = [targets[i + 1] for i in range(0, len(targets), 2)]
            new_circuit.append("H", targs)
            if pipelined:
                new_circuit.append("TICK")
            new_circuit.append("CZ", targets)
            if pipelined:
                new_circuit.append("TICK")
            new_circuit.append("H", targs)
        else:
            new_circuit.append(instruction)
    return new_circuit


def build_rotated_surface_code(
    distance: int,
    rounds: int,
    p_error: float = 0.0,
    schedule: str = "cx",
    basis: str = "Z",
) -> stim.Circuit:
    """
    Build a rotated surface code memory circuit with customizable stabilizer scheduling.

    Supports standard CX schedules as well as pipelined CZ schedules suitable for
    tunable-coupler superconducting QPUs (like IBM Heron).

    Args:
        distance: Code distance (odd integer >= 3).
        rounds: Number of syndrome extraction rounds (>= 1).
        p_error: Uniform physical error probability.
        schedule: Stabilizer schedule:
            - 'cx': standard CNOT-based syndrome extraction
            - 'cz': native CZ-based syndrome extraction
            - 'pipelined_cz': pipelined CZ layers with synchronized tick barriers
        basis: Logical memory basis ('Z' or 'X').

    Returns:
        stim.Circuit with detector annotations and logical observable.
    """
    if distance < 3 or distance % 2 == 0:
        raise ValueError(f"Distance must be odd and >= 3, got {distance}")
    if rounds < 1:
        raise ValueError(f"Rounds must be >= 1, got {rounds}")

    valid_schedules = ("cx", "cz", "pipelined_cz")
    if schedule.lower() not in valid_schedules:
        raise ValueError(f"Unknown schedule '{schedule}'. Supported: {valid_schedules}")

    task = "surface_code:rotated_memory_x" if basis.upper() == "X" else "surface_code:rotated_memory_z"

    if p_error > 0.0:
        base_circuit = stim.Circuit.generated(
            task,
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=p_error,
            before_round_data_depolarization=p_error,
            before_measure_flip_probability=p_error,
            after_reset_flip_probability=p_error,
        )
    else:
        base_circuit = stim.Circuit.generated(
            task,
            distance=distance,
            rounds=rounds,
        )

    if schedule.lower() == "cz":
        return convert_stabilizer_schedule_to_cz(base_circuit, pipelined=False)
    elif schedule.lower() == "pipelined_cz":
        return convert_stabilizer_schedule_to_cz(base_circuit, pipelined=True)
    return base_circuit


class CircuitGenerator:
    """
    Circuit generation engine.

    Generates QEC circuits via Stim and converts them to Qiskit
    circuits for real QPU execution.
    """

    def __init__(
        self,
        qec_config: QECConfig,
        noise_config: Optional[NoiseConfig] = None,
        schedule: str = "cx",
    ) -> None:
        self._qec_config = qec_config
        self._noise_config = noise_config
        self._schedule = schedule
        self._code: Optional[QECCode] = None

    @property
    def schedule(self) -> str:
        return self._schedule

    @schedule.setter
    def schedule(self, value: str) -> None:
        if value.lower() not in ("cx", "cz", "pipelined_cz"):
            raise ValueError(f"Unknown schedule: {value}")
        self._schedule = value

    @property
    def code(self) -> QECCode:
        """Get or create the QEC code instance."""
        if self._code is None:
            self._code = create_code(
                code_type=self._qec_config.code.value,
                distance=self._qec_config.distance,
                rounds=self._qec_config.rounds,
            )
        return self._code

    def generate_stim_circuit(self, schedule: Optional[str] = None) -> stim.Circuit:
        """
        Generate the Stim circuit for the configured QEC code.

        Args:
            schedule: Optional override for stabilizer schedule ('cx', 'cz', 'pipelined_cz').

        Returns:
            Stim circuit with detectors, observables, and noise.
        """
        sched = (schedule or self._schedule).lower()
        circuit = self.code.generate_circuit(noise=self._noise_config)

        if sched in ("cz", "pipelined_cz"):
            circuit = convert_stabilizer_schedule_to_cz(
                circuit,
                pipelined=(sched == "pipelined_cz"),
            )

        logger.info(
            f"Generated Stim circuit (schedule={sched}): "
            f"{circuit.num_qubits} qubits, "
            f"{circuit.num_detectors} detectors, "
            f"{circuit.num_observables} observables"
        )
        return circuit

    def get_detector_error_model(self) -> stim.DetectorErrorModel:
        """
        Get the detector error model for the circuit.

        This is the input format for PyMatching and other decoders.
        """
        circuit = self.generate_stim_circuit()
        dem = circuit.detector_error_model(decompose_errors=True)
        logger.info(f"Generated DEM with {dem.num_detectors} detectors")
        return dem

    def stim_to_qiskit(self, stim_circuit: stim.Circuit) -> Any:
        """
        Convert a Stim circuit to a Qiskit QuantumCircuit.

        This is necessary for executing QEC experiments on real IBM QPU hardware.
        The conversion preserves the logical structure while mapping to
        Qiskit's representation.

        Args:
            stim_circuit: The Stim circuit to convert.

        Returns:
            A Qiskit QuantumCircuit ready for transpilation and execution.
        """
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

        num_qubits = stim_circuit.num_qubits
        num_measurements = stim_circuit.num_measurements

        qr = QuantumRegister(num_qubits, "q")
        cr = ClassicalRegister(num_measurements, "meas")
        qc = QuantumCircuit(qr, cr)

        meas_idx = 0

        for instruction in stim_circuit.flattened():
            if not isinstance(instruction, stim.CircuitInstruction):
                continue
            name = instruction.name

            if name == "R":
                for target in instruction.targets_copy():
                    qc.reset(target.value)

            elif name == "H":
                for target in instruction.targets_copy():
                    qc.h(target.value)

            elif name == "S":
                for target in instruction.targets_copy():
                    qc.s(target.value)

            elif name == "S_DAG":
                for target in instruction.targets_copy():
                    qc.sdg(target.value)

            elif name == "X":
                for target in instruction.targets_copy():
                    qc.x(target.value)

            elif name == "Y":
                for target in instruction.targets_copy():
                    qc.y(target.value)

            elif name == "Z":
                for target in instruction.targets_copy():
                    qc.z(target.value)

            elif name in ("CX", "CNOT", "ZCX"):
                targets = instruction.targets_copy()
                for i in range(0, len(targets), 2):
                    qc.cx(targets[i].value, targets[i + 1].value)

            elif name == "CZ":
                targets = instruction.targets_copy()
                for i in range(0, len(targets), 2):
                    qc.cz(targets[i].value, targets[i + 1].value)

            elif name == "SWAP":
                targets = instruction.targets_copy()
                for i in range(0, len(targets), 2):
                    qc.swap(targets[i].value, targets[i + 1].value)

            elif name == "M" or name == "MR":
                for target in instruction.targets_copy():
                    qc.measure(target.value, meas_idx)
                    meas_idx += 1
                    if name == "MR":
                        qc.reset(target.value)

            elif name == "MX":
                for target in instruction.targets_copy():
                    qc.h(target.value)
                    qc.measure(target.value, meas_idx)
                    meas_idx += 1
                    qc.h(target.value)

            elif name == "MY":
                for target in instruction.targets_copy():
                    qc.sdg(target.value)
                    qc.h(target.value)
                    qc.measure(target.value, meas_idx)
                    meas_idx += 1
                    qc.h(target.value)
                    qc.s(target.value)

            elif name == "TICK":
                qc.barrier()

            elif name in (
                "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS",
                "SHIFT_COORDS", "REPEAT",
                "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR",
                "Z_ERROR", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2",
                "E", "ELSE_CORRELATED_ERROR",
            ):
                # Annotations and noise instructions — skip in Qiskit circuit
                pass

            else:
                logger.warning(f"Unhandled Stim instruction: {name}")

        logger.info(
            f"Converted Stim→Qiskit: {num_qubits} qubits, "
            f"{meas_idx} measurements, depth={qc.depth()}"
        )
        return qc

    def get_measurement_mapping(self, stim_circuit: stim.Circuit) -> dict[str, Any]:
        """
        Build the mapping from Stim measurement indices to
        detector/observable definitions.

        This is critical for reconstructing detection events from
        raw QPU measurement outcomes.
        """
        mapping: dict[str, Any] = {
            "num_measurements": stim_circuit.num_measurements,
            "num_detectors": stim_circuit.num_detectors,
            "num_observables": stim_circuit.num_observables,
            "detector_coords": {},
        }

        # Extract detector coordinates
        coord_dict = stim_circuit.get_detector_coordinates()
        for det_id, coords in coord_dict.items():
            mapping["detector_coords"][int(det_id)] = list(coords)

        return mapping
