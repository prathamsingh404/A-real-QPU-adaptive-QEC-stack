"""
Data provenance tracking for AdaptiveQEC.

Every numeric parameter, calibration value, and empirical factor in the
system should carry a provenance tag. This forces intellectual honesty
about what is measured vs assumed, and makes it impossible to silently
pass off hardcoded constants as experimental results.

Usage:
    from adaptive_qec.provenance import DataProvenance, ProvenanceTag

    suppression = ProvenanceTag(
        value=0.45,
        provenance=DataProvenance.ASSUMED,
        source="Estimated from IBM Orbit paper (2023), Fig. 3",
        timestamp=None,  # not from a specific measurement
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class DataProvenance(str, Enum):
    """Provenance classification for numeric parameters.

    Every number in the system should be tagged with one of these:
        MEASURED:    Direct hardware measurement at a known timestamp.
        INFERRED:    Derived/estimated from measured data (e.g., leakage
                     rate estimated from syndrome autocorrelation).
        ASSUMED:     Empirical value from literature or expert judgement.
                     Must include a citation or rationale.
        HARDCODED:   Fixed constant without justification. This is a
                     red flag — every HARDCODED value should eventually
                     become MEASURED, INFERRED, or ASSUMED with a source.
        SIMULATION:  Value obtained from a Stim or other simulator run.
    """
    MEASURED = "measured"
    INFERRED = "inferred"
    ASSUMED = "assumed"
    HARDCODED = "hardcoded"
    SIMULATION = "simulation"


@dataclass(frozen=True)
class ProvenanceTag:
    """A numeric value with provenance metadata.

    Attributes:
        value: The numeric value itself.
        provenance: How this value was obtained.
        source: Citation, experiment ID, or rationale.
        timestamp: When the value was obtained (if applicable).
        uncertainty: Estimated uncertainty (e.g., standard error).
        notes: Additional context.
    """
    value: float
    provenance: DataProvenance
    source: str = ""
    timestamp: Optional[str] = None
    uncertainty: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        d: dict[str, Any] = {
            "value": self.value,
            "provenance": self.provenance.value,
            "source": self.source,
        }
        if self.timestamp:
            d["timestamp"] = self.timestamp
        if self.uncertainty is not None:
            d["uncertainty"] = self.uncertainty
        if self.notes:
            d["notes"] = self.notes
        return d

    def __float__(self) -> float:
        return self.value

    def __repr__(self) -> str:
        return (
            f"ProvenanceTag({self.value}, {self.provenance.value}, "
            f"source={self.source!r})"
        )


@dataclass
class ProvenanceRegistry:
    """Collects all provenance tags for an experiment or module.

    Use this to audit what assumptions went into a result.
    """
    entries: dict[str, ProvenanceTag] = field(default_factory=dict)

    def register(self, name: str, tag: ProvenanceTag) -> None:
        """Register a named parameter with its provenance."""
        self.entries[name] = tag

    def get_by_provenance(self, prov: DataProvenance) -> dict[str, ProvenanceTag]:
        """Get all entries with a specific provenance type."""
        return {k: v for k, v in self.entries.items() if v.provenance == prov}

    def audit_report(self) -> str:
        """Generate a human-readable audit report."""
        lines = ["=== Provenance Audit Report ==="]
        for prov in DataProvenance:
            entries = self.get_by_provenance(prov)
            if entries:
                lines.append(f"\n[{prov.value.upper()}] ({len(entries)} parameters)")
                for name, tag in sorted(entries.items()):
                    source_str = f" — {tag.source}" if tag.source else ""
                    unc_str = f" +/- {tag.uncertainty}" if tag.uncertainty is not None else ""
                    lines.append(f"  {name}: {tag.value}{unc_str}{source_str}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return {k: v.to_dict() for k, v in self.entries.items()}
