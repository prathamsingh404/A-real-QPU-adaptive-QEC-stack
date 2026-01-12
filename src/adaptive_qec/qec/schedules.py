"""
Stabilizer measurement schedule definitions.

Defines the temporal pattern of X-type vs Z-type stabilizer measurements
in surface code QEC rounds. Under isotropic noise, a balanced 1:1
alternation (X, Z, X, Z, ...) is standard. However, real hardware
exhibits strong T1/T2 noise asymmetry:

    - When T1 << T2 (relaxation-dominated): bit-flip errors dominate,
      Z-type stabilizers (which detect X errors) should be measured
      more frequently → Z_HEAVY schedule.
    - When T2 << T1 (dephasing-dominated): phase-flip errors dominate,
      X-type stabilizers (which detect Z errors) should be measured
      more frequently → X_HEAVY schedule.

Schedule patterns are encoded as repeating sequences of 'X' and 'Z'
characters, e.g. "XZX" means 2:1 X:Z ratio per period.

References:
    - Bonilla Ataides et al., "The XZZX surface code" (2021)
    - Tuckett et al., "Tailoring Surface Codes for Highly Biased Noise"
      PRX Quantum 1, 010310 (2020)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


class ScheduleType(str, Enum):
    """Predefined stabilizer measurement schedule types."""

    BALANCED = "balanced"        # 1:1 X:Z ratio — [X, Z]
    X_HEAVY = "x_heavy"          # 2:1 X:Z ratio — [X, Z, X]
    Z_HEAVY = "z_heavy"          # 2:1 Z:X ratio — [Z, X, Z]
    EXTREME_X = "extreme_x"      # 3:1 X:Z ratio — [X, Z, X, X]
    EXTREME_Z = "extreme_z"      # 3:1 Z:X ratio — [Z, X, Z, Z]
    CUSTOM = "custom"            # User-defined pattern


@dataclass(frozen=True)
class StabilizerSchedule:
    """
    A repeating measurement schedule for X and Z stabilizers.

    Attributes:
        pattern: Repeating sequence of 'X' and 'Z' characters.
                 E.g. "XZ" for balanced, "XZX" for 2:1 X-heavy.
        schedule_type: Classification of the schedule.
        x_fraction: Fraction of rounds measuring X stabilizers.
        z_fraction: Fraction of rounds measuring Z stabilizers.
    """

    pattern: str
    schedule_type: ScheduleType

    def __post_init__(self) -> None:
        if not self.pattern:
            raise ValueError("Schedule pattern must be non-empty")
        invalid = set(self.pattern) - {"X", "Z"}
        if invalid:
            raise ValueError(
                f"Schedule pattern must contain only 'X' and 'Z', "
                f"got invalid characters: {invalid}"
            )

    @property
    def x_fraction(self) -> float:
        """Fraction of rounds that measure X-type stabilizers."""
        return self.pattern.count("X") / len(self.pattern)

    @property
    def z_fraction(self) -> float:
        """Fraction of rounds that measure Z-type stabilizers."""
        return self.pattern.count("Z") / len(self.pattern)

    @property
    def period(self) -> int:
        """Number of rounds before the schedule repeats."""
