"""
Realistic noise injection engine.

For controlled benchmark experiments, injects:
    - Independent noise: X, Y, Z, depolarizing
    - Readout noise: P(0|1), P(1|0)
    - Correlated noise: two-qubit, spatial, temporal
    - Leakage, crosstalk, drift

Goal: synthetic → controlled benchmark, real QPU → reality.
Do not confuse the two.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import stim

from adaptive_qec.config import NoiseConfig

logger = logging.getLogger(__name__)


class NoiseInjector:
    """
    Injects controlled noise into Stim circuits for benchmarking.

    This is NOT simulation — it's a controlled benchmark comparator.
    The real QPU data is always the ground truth.
    """

    def __init__(self, noise_config: NoiseConfig) -> None:
        self._config = noise_config

    def inject_independent_noise(
        self,
        circuit: stim.Circuit,
        p_x: float = 0.0,
        p_y: float = 0.0,
        p_z: float = 0.0,
    ) -> stim.Circuit:
        """
        Inject independent Pauli noise after each gate.

        For depolarizing: p_x = p_y = p_z = p/3
        """
        noisy = stim.Circuit()

        for instruction in circuit.flattened():
            noisy.append(instruction)

            if instruction.name in ("H", "S", "S_DAG", "X", "Y", "Z"):
                targets = instruction.targets_copy()
                qubit_targets = [t.value for t in targets]
                if p_x + p_y + p_z > 0:
                    noisy.append(
                        "PAULI_CHANNEL_1",
                        qubit_targets,
                        [p_x, p_y, p_z],
                    )
            elif instruction.name in ("CX", "CNOT", "CZ", "SWAP"):
                targets = instruction.targets_copy()
