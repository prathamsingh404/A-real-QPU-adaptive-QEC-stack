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
