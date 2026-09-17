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
from typing import Optional

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
