"""
QEC Code Geometry and Stabilizer Mapping (src/adaptive_qec/qec/geometry.py).

Provides code-aware spatial and stabilizer mapping for topological codes.
Replaces naive array bisections (e.g. `syndromes[:, :n//2]`) with mathematically
rigorous stabilizer identification (X vs Z checks) based on physical coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import stim


@dataclass(frozen=True)
class DetectorInfo:
    """Detailed spatial, temporal, and algebraic metadata for a single detector."""
    detector_id: int
    stabilizer_type: Literal["X", "Z", "UNKNOWN"]
    x: float
    y: float
    round: int
    boundary: Optional[str] = None


@dataclass
class DetectorMap:
    """Map of all detectors in a QEC circuit to their physical coordinates and check types."""
    detectors: dict[int, DetectorInfo] = field(default_factory=dict)
    x_indices: list[int] = field(default_factory=list)
    z_indices: list[int] = field(default_factory=list)

    @classmethod
    def from_stim_circuit(
        cls,
        circuit: stim.Circuit,
        code_type: str = "surface_code:rotated_memory_z",
    ) -> DetectorMap:
        """Parse detector coordinates directly from a Stim circuit.

        For rotated surface codes in Stim (rotated_memory_z):
        - Detectors at coordinate (x, y) with (x + y)//2 % 2 == 0 correspond to Z stabilizers.
        - Detectors at coordinate (x, y) with (x + y)//2 % 2 == 1 correspond to X stabilizers.
        """
        coords_dict = circuit.get_detector_coordinates()
        detectors: dict[int, DetectorInfo] = {}
        x_indices: list[int] = []
        z_indices: list[int] = []

        for d_id, coords in coords_dict.items():
            x = float(coords[0]) if len(coords) > 0 else 0.0
            y = float(coords[1]) if len(coords) > 1 else 0.0
            t = int(coords[2]) if len(coords) > 2 else 0

            # Determine stabilizer type
            if "rotated" in code_type or "surface" in code_type:
                parity = int(round((x + y) / 2.0)) % 2
                stab_type: Literal["X", "Z", "UNKNOWN"] = "Z" if parity == 0 else "X"
            elif "repetition" in code_type:
                stab_type = "Z"
            else:
                stab_type = "UNKNOWN"

            if stab_type == "X":
                x_indices.append(d_id)
            elif stab_type == "Z":
                z_indices.append(d_id)

            detectors[d_id] = DetectorInfo(
                detector_id=d_id,
                stabilizer_type=stab_type,
                x=x,
                y=y,
                round=t,
            )

        return cls(
            detectors=detectors,
            x_indices=sorted(x_indices),
            z_indices=sorted(z_indices),
        )

    def detectors_of_type(self, stab_type: Literal["X", "Z"]) -> list[int]:
        """Return all detector indices corresponding to a specific check basis."""
        if stab_type == "X":
            return list(self.x_indices)
        elif stab_type == "Z":
            return list(self.z_indices)
        return []

    def extract_defect_counts(self, syndromes: np.ndarray) -> tuple[int, int]:
        """Extract exact total (X_defects, Z_defects) from a 2D syndrome batch.

        Args:
            syndromes: Array of shape (shots, num_detectors).

        Returns:
            Tuple of (total_x_defects, total_z_defects) summed across all shots.
        """
        if syndromes.size == 0 or len(self.detectors) == 0:
            return 0, 0

        # Filter indices within syndrome array bounds
        valid_x = [i for i in self.x_indices if i < syndromes.shape[1]]
        valid_z = [i for i in self.z_indices if i < syndromes.shape[1]]

        x_defects = int(np.sum(syndromes[:, valid_x])) if valid_x else 0
        z_defects = int(np.sum(syndromes[:, valid_z])) if valid_z else 0

        return x_defects, z_defects


@dataclass
class CodeGeometry:
    """High-level geometric representation of a topological code patch."""
    code_name: str
    distance: int
    rounds: int
    detector_map: DetectorMap
    num_data_qubits: int
    num_ancilla_qubits: int

    @classmethod
    def for_rotated_surface_code(cls, distance: int, rounds: int) -> CodeGeometry:
        """Construct standard geometry for a rotated surface code patch."""
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=distance,
            rounds=rounds,
        )
        det_map = DetectorMap.from_stim_circuit(circuit, "surface_code:rotated_memory_z")
        return cls(
            code_name="surface_code:rotated_memory_z",
            distance=distance,
            rounds=rounds,
            detector_map=det_map,
            num_data_qubits=distance * distance,
            num_ancilla_qubits=distance * distance - 1,
        )
