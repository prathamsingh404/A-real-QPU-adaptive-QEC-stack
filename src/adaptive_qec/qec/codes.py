"""
QEC code definitions and class hierarchy.

Defines quantum error-correcting codes with their Stim circuit representations,
stabilizer generators, logical operators, and coordinate mappings.
Supports repetition code, rotated surface code, and 2D color code with
parameterizable distance and rounds.

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


class QuantumCode(ABC):
    """
    Abstract base class for quantum error-correcting codes.

    Enforces uniform interfaces for stabilizer generators, logical operators,
    coordinate mappings, and circuit generation across code families.
    """

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

    @property
    @abstractmethod
    def stabilizer_generators(self) -> list[str]:
        """List of stabilizer generator Pauli string representations."""
        ...

    @property
    def stabilizers(self) -> list[str]:
        """Alias for stabilizer_generators."""
        return self.stabilizer_generators

    @property
    @abstractmethod
    def logical_operators(self) -> dict[str, str]:
        """Dictionary of logical operators, e.g. {'X': ..., 'Z': ...}."""
        ...

    @property
    def logical_x(self) -> str:
        """Logical X operator."""
        return self.logical_operators.get("X", "")

    @property
    def logical_z(self) -> str:
        """Logical Z operator."""
        return self.logical_operators.get("Z", "")

    @property
    @abstractmethod
    def coordinate_mappings(self) -> dict[int, tuple[float, float]]:
        """Mapping from qubit index to 2D (x, y) spatial coordinates."""
        ...


# Backward compatibility alias
QECCode = QuantumCode


class RepetitionCode(QuantumCode):
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

    @property
    def coordinate_mappings(self) -> dict[int, tuple[float, float]]:
        """Mapping of qubit index to (x, y) coordinates."""
        coords: dict[int, tuple[float, float]] = {}
        for i in range(self.num_data):
            coords[i] = (float(2 * i), 0.0)
        for i in range(self.num_ancilla):
            coords[self.num_data + i] = (float(2 * i + 1), 0.0)
        return coords

    @property
    def stabilizer_generators(self) -> list[str]:
        """Stabilizer generators: Z_i Z_{i+1} across adjacent data qubits."""
        return [f"Z{i} Z{i+1}" for i in range(self.distance - 1)]

    @property
    def logical_operators(self) -> dict[str, str]:
        """Logical operators for the repetition code."""
        return {
            "X": " ".join(f"X{i}" for i in range(self.distance)),
            "Z": "Z0",
        }

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


class SurfaceCode(QuantumCode):
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

    @property
    def coordinate_mappings(self) -> dict[int, tuple[float, float]]:
        """Qubit coordinates extracted from the generated Stim circuit."""
        circuit = self.generate_circuit()
        coords: dict[int, tuple[float, float]] = {}
        for inst in circuit.flattened():
            if inst.name == "QUBIT_COORDS":
                q = inst.targets_copy()[0].value
                args = inst.gate_args_copy()
                coords[q] = (float(args[0]), float(args[1])) if len(args) >= 2 else (float(args[0]), 0.0)
        return coords

    @property
    def stabilizer_generators(self) -> list[str]:
        """
        Stabilizer generators for the rotated surface code.
        Extracted from the syndrome extraction round interactions.
        """
        circuit = self.generate_circuit()
        h_qubits = set()
        interactions: dict[int, tuple[str, set[int]]] = {}
        for inst in circuit.flattened():
            if inst.name == "H":
                for t in inst.targets_copy():
                    h_qubits.add(t.value)
            elif inst.name == "CX":
                targets = inst.targets_copy()
                for i in range(0, len(targets), 2):
                    c_q = targets[i].value
                    t_q = targets[i + 1].value
                    if c_q in h_qubits:
                        interactions.setdefault(c_q, ("X", set()))[1].add(t_q)
                    else:
                        interactions.setdefault(t_q, ("Z", set()))[1].add(c_q)
            elif inst.name in ("M", "MR"):
                if interactions:
                    break

        generators: list[str] = []
        for ancilla, (stype, data_qubits) in sorted(interactions.items()):
            terms = " ".join(f"{stype}{d}" for d in sorted(data_qubits))
            generators.append(terms)
        return generators

    @property
    def logical_operators(self) -> dict[str, str]:
        """Logical operators along boundaries for rotated surface code."""
        d = self.distance
        coords = self.coordinate_mappings
        circuit = self.generate_circuit()
        meas_qubits = []
        for inst in circuit.flattened():
            if inst.name == "M":
                meas_qubits = [t.value for t in inst.targets_copy()]
        data_qubits = sorted(meas_qubits) if len(meas_qubits) == d * d else sorted(list(coords.keys())[:d * d])

        data_coords = {q: coords[q] for q in data_qubits if q in coords}
        sorted_by_y = sorted(data_coords.items(), key=lambda item: (item[1][1], item[1][0]))
        sorted_by_x = sorted(data_coords.items(), key=lambda item: (item[1][0], item[1][1]))

        x_boundary = [item[0] for item in sorted_by_y[:d]]
        z_boundary = [item[0] for item in sorted_by_x[:d]]

        return {
            "X": " ".join(f"X{q}" for q in x_boundary),
            "Z": " ".join(f"Z{q}" for q in z_boundary),
        }

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
        num_ancilla = (d * d - 1)
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
        coord_dict = circuit.get_detector_coordinates()
        result = np.zeros((len(coord_dict), 3))
        for det_id, coord in coord_dict.items():
            if det_id < len(result):
                result[det_id, :len(coord)] = coord[:3]
        return result


class UnrotatedSurfaceCode(QuantumCode):
    """Unrotated (standard planar) surface code implementation."""

    def __init__(self, distance: int, rounds: int) -> None:
        if distance < 3 or distance % 2 == 0:
            raise ValueError(f"Distance must be an odd integer >= 3, got {distance}")
        if rounds < 1:
            raise ValueError(f"Rounds must be >= 1, got {rounds}")
        self.distance = distance
        self.rounds = rounds

    @property
    def coordinate_mappings(self) -> dict[int, tuple[float, float]]:
        """Qubit coordinates from Stim circuit."""
        circuit = self.generate_circuit()
        coords: dict[int, tuple[float, float]] = {}
        for inst in circuit.flattened():
            if inst.name == "QUBIT_COORDS":
                q = inst.targets_copy()[0].value
                args = inst.gate_args_copy()
                coords[q] = (float(args[0]), float(args[1])) if len(args) >= 2 else (float(args[0]), 0.0)
        return coords

    @property
    def stabilizer_generators(self) -> list[str]:
        """Stabilizer generators for unrotated surface code."""
        d = self.distance
        num_ancilla = 2 * d * (d - 1)
        return [f"S_{i}" for i in range(num_ancilla)]

    @property
    def logical_operators(self) -> dict[str, str]:
        """Logical operators for unrotated surface code."""
        d = self.distance
        return {
            "X": " ".join(f"X{q}" for q in range(d)),
            "Z": " ".join(f"Z{q * d}" for q in range(d)),
        }

    def generate_circuit(
        self,
        noise: Optional[NoiseConfig] = None,
        embedding: Optional[Any] = None,
    ) -> stim.Circuit:
        p = noise.gate.two_qubit if noise and noise.gate else 0.0
        if p > 0:
            return stim.Circuit.generated(
                "surface_code:unrotated_memory_z",
                distance=self.distance,
                rounds=self.rounds,
                after_clifford_depolarization=p,
                before_round_data_depolarization=p,
                before_measure_flip_probability=p,
                after_reset_flip_probability=p,
            )
        return stim.Circuit.generated(
            "surface_code:unrotated_memory_z",
            distance=self.distance,
            rounds=self.rounds,
        )

    def get_info(self) -> CodeInfo:
        d = self.distance
        num_data = d * d + (d - 1) * (d - 1)
        num_ancilla = 2 * d * (d - 1)
        return CodeInfo(
            name="unrotated_surface",
            distance=d,
            rounds=self.rounds,
            num_data_qubits=num_data,
            num_ancilla_qubits=num_ancilla,
            num_detectors=num_ancilla * self.rounds,
            num_observables=1,
        )

    def get_detector_coordinates(self) -> np.ndarray:
        circuit = self.generate_circuit()
        coord_dict = circuit.get_detector_coordinates()
        result = np.zeros((len(coord_dict), 3))
        for det_id, coord in coord_dict.items():
            if det_id < len(result):
                result[det_id, :len(coord)] = coord[:3]
        return result


class ColorCode(QuantumCode):
    """
    2D Triangular Color Code (Steane [[7, 1, 3]] for d=3).

    Color codes feature transversal Clifford gates (H, S, CNOT) and
    measure both X and Z stabilizers on each 2D face/plaquette.
    """

    def __init__(self, distance: int = 3, rounds: int = 2) -> None:
        if distance < 3 or distance % 2 == 0:
            raise ValueError(f"Distance must be odd and >= 3, got {distance}")
        if rounds < 2:
            raise ValueError(f"Color code XYZ memory requires rounds >= 2, got {rounds}")

        self.distance = distance
        self.rounds = rounds
        # Triangular color code: n = (3 * d^2 + 1) / 4 data qubits
        self.num_data = (3 * distance ** 2 + 1) // 4
        self.num_faces = (self.num_data - 1) // 2
        self.num_ancilla = self.num_faces

    @property
    def coordinate_mappings(self) -> dict[int, tuple[float, float]]:
        """Qubit coordinates from Stim circuit."""
        circuit = self.generate_circuit()
        coords: dict[int, tuple[float, float]] = {}
        for inst in circuit.flattened():
            if inst.name == "QUBIT_COORDS":
                q = inst.targets_copy()[0].value
                args = inst.gate_args_copy()
                coords[q] = (float(args[0]), float(args[1])) if len(args) >= 2 else (float(args[0]), 0.0)
        return coords

    @property
    def stabilizer_generators(self) -> list[str]:
        """
        Stabilizer generators for color code.
        Each face supports both an X and a Z stabilizer check.
        """
        if self.distance == 3:
            faces = [
                [0, 1, 4, 5],
                [1, 2, 5, 6],
                [4, 5, 7, 8] if 7 in self.coordinate_mappings else [0, 2, 4, 6],
            ]
            stabs = []
            for face in faces:
                stabs.append(" ".join(f"X{q}" for q in face))
                stabs.append(" ".join(f"Z{q}" for q in face))
            return stabs

        # General distance face checks
        stabs = []
        for f in range(self.num_faces):
            stabs.append(f"X_face_{f}")
            stabs.append(f"Z_face_{f}")
        return stabs

    @property
    def logical_operators(self) -> dict[str, str]:
        """Logical operators: transversal along boundary or transversal all data."""
        boundary = list(range(self.distance))
        return {
            "X": " ".join(f"X{q}" for q in boundary),
            "Z": " ".join(f"Z{q}" for q in boundary),
        }

    def generate_circuit(
        self,
        noise: Optional[NoiseConfig] = None,
        embedding: Optional[Any] = None,
    ) -> stim.Circuit:
        """Generate Stim circuit for color code memory."""
        p = noise.gate.two_qubit if noise and noise.gate else 0.0

        if p > 0.0:
            circuit = stim.Circuit.generated(
                "color_code:memory_xyz",
                distance=self.distance,
                rounds=self.rounds,
                after_clifford_depolarization=p,
                before_round_data_depolarization=p,
                before_measure_flip_probability=p,
                after_reset_flip_probability=p,
            )
        else:
            circuit = stim.Circuit.generated(
                "color_code:memory_xyz",
                distance=self.distance,
                rounds=self.rounds,
            )

        if embedding is not None and hasattr(embedding, "swap_count"):
            swap_count = embedding.swap_count()
            if swap_count > 0 and p > 0:
                extra_p = min(0.5, p * swap_count * 0.05)
                circuit.append("DEPOLARIZE1", list(range(circuit.num_qubits)), [extra_p])

        logger.info(
            f"Generated color code circuit: d={self.distance}, R={self.rounds}, "
            f"qubits={circuit.num_qubits}, detectors={circuit.num_detectors}"
        )
        return circuit

    def get_info(self) -> CodeInfo:
        """Get color code static info."""
        circuit = self.generate_circuit()
        return CodeInfo(
            name="color",
            distance=self.distance,
            rounds=self.rounds,
            num_data_qubits=self.num_data,
            num_ancilla_qubits=self.num_ancilla,
            num_detectors=circuit.num_detectors,
            num_observables=circuit.num_observables,
        )

    def get_detector_coordinates(self) -> np.ndarray:
        """Get detector coordinates from Stim circuit."""
        circuit = self.generate_circuit()
        coord_dict = circuit.get_detector_coordinates()
        result = np.zeros((len(coord_dict), 3))
        for det_id, coord in coord_dict.items():
            if det_id < len(result):
                result[det_id, :len(coord)] = coord[:3]
        return result


def create_code(code_type: str, distance: int, rounds: int) -> QuantumCode:
    """
    Factory function to create a QEC code.

    Args:
        code_type: "repetition", "surface", "unrotated_surface", or "color"
        distance: Code distance (must be odd, >= 3)
        rounds: Number of QEC rounds

    Returns:
        QuantumCode instance.
    """
    codes: dict[str, type[QuantumCode]] = {
        "repetition": RepetitionCode,
        "surface": SurfaceCode,
        "unrotated_surface": UnrotatedSurfaceCode,
        "color": ColorCode,
    }

    norm_type = str(code_type).lower().split(".")[-1]
    if norm_type not in codes:
        raise ValueError(
            f"Unknown code type '{code_type}'. Available: {list(codes.keys())}"
        )

    return codes[norm_type](distance=distance, rounds=rounds)


# Backwards compatibility alias
QECCode = QuantumCode
