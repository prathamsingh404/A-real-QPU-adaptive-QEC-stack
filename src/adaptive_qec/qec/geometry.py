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
