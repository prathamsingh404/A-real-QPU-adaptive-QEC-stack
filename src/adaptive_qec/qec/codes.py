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
