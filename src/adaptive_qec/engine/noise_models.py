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
                qubit_pairs = [(targets[i].value, targets[i + 1].value)
                               for i in range(0, len(targets), 2)]
                if p_x + p_y + p_z > 0:
                    for q1, q2 in qubit_pairs:
                        noisy.append("DEPOLARIZE2", [q1, q2], [p_x + p_y + p_z])

        return noisy

    def inject_readout_noise(
        self,
        circuit: stim.Circuit,
        p0_given_1: Optional[float] = None,
        p1_given_0: Optional[float] = None,
    ) -> stim.Circuit:
        """
        Inject readout noise before measurements.

        p0_given_1: probability of reading 0 when state is 1
        p1_given_0: probability of reading 1 when state is 0
        """
        p01 = p0_given_1 if p0_given_1 is not None else self._config.readout.p0_given_1
        p10 = p1_given_0 if p1_given_0 is not None else self._config.readout.p1_given_0

        # Symmetric readout noise as X_ERROR before measurement
        p_flip = (p01 + p10) / 2

        noisy = stim.Circuit()
        for instruction in circuit.flattened():
            if instruction.name in ("M", "MR", "MX", "MY") and p_flip > 0:
                targets = instruction.targets_copy()
                qubit_targets = [t.value for t in targets]
                noisy.append("X_ERROR", qubit_targets, [p_flip])
            noisy.append(instruction)

        return noisy

    def inject_correlated_noise(
        self,
        circuit: stim.Circuit,
        correlation_pairs: list[tuple[int, int]],
        strength: Optional[float] = None,
    ) -> stim.Circuit:
        """
        Inject correlated two-qubit noise on specified pairs.

        This models spatial correlations between neighboring qubits.
        """
        p = strength if strength is not None else self._config.correlated.strength

        noisy = stim.Circuit()
        for instruction in circuit.flattened():
            noisy.append(instruction)

            if instruction.name == "TICK" and p > 0:
                for q1, q2 in correlation_pairs:
                    noisy.append("DEPOLARIZE2", [q1, q2], [p])

        return noisy

    def inject_drift(
        self,
        circuit: stim.Circuit,
        base_error: float,
        drift_rate: float,
        time_step: float = 1.0,
