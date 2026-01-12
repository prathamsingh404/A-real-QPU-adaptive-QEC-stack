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
        return len(self.pattern)

    @property
    def ratio_xz(self) -> float:
        """X:Z ratio. Returns float('inf') if no Z rounds."""
        z_count = self.pattern.count("Z")
        if z_count == 0:
            return float("inf")
        return self.pattern.count("X") / z_count

    def round_type(self, round_idx: int) -> str:
        """
        Get the stabilizer type for a given round index.

        Args:
            round_idx: Zero-based round index.

        Returns:
            'X' or 'Z'
        """
        return self.pattern[round_idx % len(self.pattern)]

    def generate_sequence(self, n_rounds: int) -> list[str]:
        """
        Generate a concrete sequence of round types.

        Args:
            n_rounds: Number of QEC rounds.

        Returns:
            List of 'X' and 'Z' strings, one per round.
        """
        return [self.round_type(i) for i in range(n_rounds)]

    def iter_rounds(self) -> Iterator[str]:
        """Infinite iterator over round types."""
        idx = 0
        while True:
            yield self.pattern[idx % len(self.pattern)]
            idx += 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pattern": self.pattern,
            "schedule_type": self.schedule_type.value,
            "x_fraction": round(self.x_fraction, 4),
            "z_fraction": round(self.z_fraction, 4),
            "period": self.period,
            "ratio_xz": round(self.ratio_xz, 4)
            if self.ratio_xz != float("inf")
            else "inf",
        }

    def __repr__(self) -> str:
        return (
            f"StabilizerSchedule(pattern='{self.pattern}', "
            f"type={self.schedule_type.value}, "
            f"X:Z={self.ratio_xz:.2f})"
        )


# -----------------------------------------------------------------------
# Predefined schedule library
# -----------------------------------------------------------------------

PREDEFINED_SCHEDULES: dict[ScheduleType, StabilizerSchedule] = {
    ScheduleType.BALANCED: StabilizerSchedule(
        pattern="XZ",
        schedule_type=ScheduleType.BALANCED,
    ),
    ScheduleType.X_HEAVY: StabilizerSchedule(
        pattern="XZX",
        schedule_type=ScheduleType.X_HEAVY,
    ),
    ScheduleType.Z_HEAVY: StabilizerSchedule(
        pattern="ZXZ",
        schedule_type=ScheduleType.Z_HEAVY,
    ),
    ScheduleType.EXTREME_X: StabilizerSchedule(
        pattern="XZXX",
        schedule_type=ScheduleType.EXTREME_X,
    ),
    ScheduleType.EXTREME_Z: StabilizerSchedule(
        pattern="ZXZZ",
