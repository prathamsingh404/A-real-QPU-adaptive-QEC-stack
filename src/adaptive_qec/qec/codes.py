"""
QEC code definitions.

Defines quantum error-correcting codes with their Stim circuit representations.
Supports repetition code and rotated surface code with parameterizable
distance and rounds.

The generated Stim circuits include:
- Full stabilizer measurement circuits
- Detector annotations
- Observable annotations
- Configurable noise injection points
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import stim

from adaptive_qec.config import NoiseConfig

logger = logging.getLogger(__name__)


@dataclass
class CodeInfo:
    """Static information about a QEC code instance."""
    name: str
    distance: int
    rounds: int
    num_data_qubits: int
    num_ancilla_qubits: int
    num_detectors: int
    num_observables: int


class QECCode(ABC):
    """Abstract base for quantum error-correcting codes."""

    @abstractmethod
    def generate_circuit(
        self,
        noise: Optional[NoiseConfig] = None,
        embedding: Optional[Any] = None,
    ) -> stim.Circuit:
        """
        Generate the Stim circuit for this code.

        Args:
            noise: Optional noise configuration to inject.
            embedding: Optional hardware embedding mapping logical to physical qubits.

        Returns:
            Stim circuit with detectors and observables annotated.
        """
        ...

    @abstractmethod
    def get_info(self) -> CodeInfo:
        """Get static information about this code instance."""
        ...

    @abstractmethod
    def get_detector_coordinates(self) -> np.ndarray:
        """Get (x, y, t) coordinates for each detector."""
        ...


class RepetitionCode(QECCode):
    """
    Repetition code for bit-flip errors.

    The simplest QEC code — good for initial testing and validation.
    d data qubits, d-1 ancilla qubits measuring ZZ stabilizers.
    """

    def __init__(self, distance: int, rounds: int) -> None:
        if distance < 3 or distance % 2 == 0:
            raise ValueError(f"Distance must be odd and >= 3, got {distance}")
        if rounds < 1:
            raise ValueError(f"Rounds must be >= 1, got {rounds}")

        self.distance = distance
        self.rounds = rounds
        self.num_data = distance
        self.num_ancilla = distance - 1

    def generate_circuit(
        self,
        noise: Optional[NoiseConfig] = None,
        embedding: Optional[Any] = None,
    ) -> stim.Circuit:
        """Generate repetition code circuit with Stim."""
        circuit = stim.Circuit()

        d = self.distance
        num_data = self.num_data
        num_ancilla = self.num_ancilla

        # Qubit layout: data qubits 0..d-1, ancilla qubits d..2d-2
        data_qubits = list(range(num_data))
        ancilla_qubits = list(range(num_data, num_data + num_ancilla))

        # Assign coordinates for visualization
        for i, q in enumerate(data_qubits):
            circuit.append("QUBIT_COORDS", [q], [2 * i, 0])
        for i, q in enumerate(ancilla_qubits):
            circuit.append("QUBIT_COORDS", [q], [2 * i + 1, 0])

        # Initialize data qubits
        circuit.append("R", data_qubits)

        # Noise parameters
        p1 = noise.gate.single_qubit if noise else 0.0
        p2 = noise.gate.two_qubit if noise else 0.0
        p_ro = (noise.readout.p0_given_1 + noise.readout.p1_given_0) / 2 if noise else 0.0

        # QEC rounds
        for r in range(self.rounds):
            # Reset ancillas
            circuit.append("R", ancilla_qubits)

            # Apply CNOT gates for ZZ stabilizer measurements
            for i in range(num_ancilla):
                circuit.append("CNOT", [data_qubits[i], ancilla_qubits[i]])
                if p2 > 0:
                    circuit.append("DEPOLARIZE2", [data_qubits[i], ancilla_qubits[i]], [p2])

            for i in range(num_ancilla):
                circuit.append("CNOT", [data_qubits[i + 1], ancilla_qubits[i]])
                if p2 > 0:
                    circuit.append("DEPOLARIZE2", [data_qubits[i + 1], ancilla_qubits[i]], [p2])

            # Measure ancillas
            if p_ro > 0:
                circuit.append("X_ERROR", ancilla_qubits, [p_ro])
            circuit.append("M", ancilla_qubits)

            # Detectors: compare current measurement to previous
            for i in range(num_ancilla):
                if r == 0:
                    # First round: detector is just the measurement
                    circuit.append(
                        "DETECTOR",
                        [stim.target_rec(-(num_ancilla - i))],
                        [2 * i + 1, 0, r],
                    )
                else:
                    # Subsequent rounds: compare to previous round
                    circuit.append(
                        "DETECTOR",
                        [
                            stim.target_rec(-(num_ancilla - i)),
                            stim.target_rec(-(2 * num_ancilla - i)),
                        ],
                        [2 * i + 1, 0, r],
                    )

            circuit.append("SHIFT_COORDS", [], [0, 0, 1])

        # Final data measurement
        if p_ro > 0:
            circuit.append("X_ERROR", data_qubits, [p_ro])
        circuit.append("M", data_qubits)

        # Final detectors: compare data measurements to last round of ancillas
        for i in range(num_ancilla):
            circuit.append(
                "DETECTOR",
                [
                    stim.target_rec(-(num_data - i)),
                    stim.target_rec(-(num_data - i - 1)),
                    stim.target_rec(-(num_data + num_ancilla - i)),
                ],
                [2 * i + 1, 0, self.rounds],
            )

        # Observable: parity of all data qubits
        obs_targets = [stim.target_rec(-(num_data - i)) for i in range(num_data)]
        circuit.append("OBSERVABLE_INCLUDE", obs_targets, [0])

        logger.info(
            f"Generated repetition code circuit: d={d}, R={self.rounds}, "
            f"detectors={circuit.num_detectors}, observables={circuit.num_observables}"
        )
        return circuit

    def get_info(self) -> CodeInfo:
        """Get repetition code info."""
        return CodeInfo(
            name="repetition",
            distance=self.distance,
            rounds=self.rounds,
            num_data_qubits=self.num_data,
            num_ancilla_qubits=self.num_ancilla,
            num_detectors=self.num_ancilla * (self.rounds + 1),
            num_observables=1,
        )

    def get_detector_coordinates(self) -> np.ndarray:
        """Get detector coordinates (x, y, t)."""
        coords = []
        for r in range(self.rounds + 1):
            for i in range(self.num_ancilla):
                coords.append([2 * i + 1, 0, r])
        return np.array(coords)


class SurfaceCode(QECCode):
    """
    Rotated surface code.

    The workhorse code for near-term QEC. Uses Stim's built-in
    surface code circuit generation for correctness, with custom
    noise injection.
    """

    def __init__(self, distance: int, rounds: int) -> None:
        if distance < 3 or distance % 2 == 0:
            raise ValueError(f"Distance must be odd and >= 3, got {distance}")
        if rounds < 1:
            raise ValueError(f"Rounds must be >= 1, got {rounds}")

        self.distance = distance
        self.rounds = rounds

    def generate_circuit(
        self,
        noise: Optional[NoiseConfig] = None,
        embedding: Optional[Any] = None,
    ) -> stim.Circuit:
        """
        Generate rotated surface code circuit.

        Uses Stim's generated circuit with appropriate noise parameters.
        If an embedding is provided, accounts for hardware topology mapping and
        SWAP routing overhead.
        """
        # Determine noise level for Stim's circuit generation
        p = 0.0
        if noise:
            # Use the two-qubit gate error as the primary noise parameter
            p = noise.gate.two_qubit

        if p > 0:
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                distance=self.distance,
                rounds=self.rounds,
                after_clifford_depolarization=p,
                before_round_data_depolarization=p,
                before_measure_flip_probability=p,
                after_reset_flip_probability=p,
            )
        else:
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                distance=self.distance,
                rounds=self.rounds,
            )

        if embedding is not None and hasattr(embedding, "swap_count"):
            swap_count = embedding.swap_count()
            if swap_count > 0 and p > 0:
                extra_p = min(0.5, p * swap_count * 0.05)
                circuit.append("DEPOLARIZE1", list(range(self.distance ** 2)), [extra_p])

        logger.info(
            f"Generated surface code circuit: d={self.distance}, R={self.rounds}, "
            f"detectors={circuit.num_detectors}, observables={circuit.num_observables}, "
            f"noise_p={p:.6f}, embedded={embedding is not None}"
        )
        return circuit

    def get_info(self) -> CodeInfo:
        """Get surface code info."""
        d = self.distance
        num_data = d * d
        num_ancilla = (d * d - 1)  # approximate for rotated code
        # Detector count: ancillas * (rounds + 1) approximately
        # Exact count comes from the Stim circuit
        return CodeInfo(
            name="surface",
            distance=d,
            rounds=self.rounds,
            num_data_qubits=num_data,
            num_ancilla_qubits=num_ancilla,
            num_detectors=(d * d - 1) * self.rounds + (d * d - 1) // 2,
            num_observables=1,
        )

    def get_detector_coordinates(self) -> np.ndarray:
        """
        Get detector coordinates from the generated Stim circuit.

        Returns coordinates from the circuit itself for accuracy.
        """
        circuit = self.generate_circuit()
        # Use Stim's built-in coordinate extraction
        coord_dict = circuit.get_detector_coordinates()
        result = np.zeros((len(coord_dict), 3))
        for det_id, coord in coord_dict.items():
            if det_id < len(result):
                result[det_id, :len(coord)] = coord[:3]
        return result


def create_code(code_type: str, distance: int, rounds: int) -> QECCode:
    """
    Factory function to create a QEC code.

    Args:
        code_type: "repetition" or "surface"
        distance: Code distance (must be odd, >= 3)
        rounds: Number of QEC rounds

    Returns:
        QECCode instance.
    """
    codes = {
        "repetition": RepetitionCode,
        "surface": SurfaceCode,
    }

    if code_type not in codes:
        raise ValueError(
            f"Unknown code type '{code_type}'. Available: {list(codes.keys())}"
        )

    return codes[code_type](distance=distance, rounds=rounds)
