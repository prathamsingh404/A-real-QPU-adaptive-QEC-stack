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


class CircuitGenerator:
    """
    Circuit generation engine.

    Generates QEC circuits via Stim and converts them to Qiskit
    circuits for real QPU execution.
    """

    def __init__(self, qec_config: QECConfig, noise_config: Optional[NoiseConfig] = None) -> None:
        self._qec_config = qec_config
        self._noise_config = noise_config
        self._code: Optional[QECCode] = None

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

    def generate_stim_circuit(self) -> stim.Circuit:
        """
        Generate the Stim circuit for the configured QEC code.

        Returns:
            Stim circuit with detectors, observables, and noise.
        """
        circuit = self.code.generate_circuit(noise=self._noise_config)
        logger.info(
            f"Generated Stim circuit: "
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
            mapping["detector_coords"][int(det_id)] = coords.tolist()

        return mapping
