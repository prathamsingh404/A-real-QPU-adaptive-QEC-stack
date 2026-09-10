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
