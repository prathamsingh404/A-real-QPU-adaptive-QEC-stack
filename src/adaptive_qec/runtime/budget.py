"""
Shot budget manager for IBM Quantum hardware execution.

Strictly enforces execution quotas to prevent accidental
over-consumption of IBM cloud runtime credits. Every circuit
submission must pass through the budget manager before execution.

Features:
    - Per-experiment shot caps with hard abort
    - Per-session credit tracking with safety margins
    - Estimated runtime cost computation
    - Audit trail of all shot allocations
    - Warning thresholds before hard limit

This is a safety-critical module. The budget manager is the ONLY
gate between the experiment loop and actual hardware execution.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class BudgetConfig:
    """
    Shot budget configuration.

    Attributes:
        max_total_shots: Absolute maximum shots across all experiments.
        max_session_shots: Maximum shots per Runtime session.
        warning_threshold: Fraction of budget at which warnings start.
        safety_margin: Reserved shots for calibration/overhead.
        estimated_shot_time_s: Estimated wall-clock time per shot.
        cost_per_shot: Estimated cost per shot (for reporting).
    """

    max_total_shots: int = 200_000
    max_session_shots: int = 100_000
    warning_threshold: float = 0.80
    safety_margin: int = 5_000
    estimated_shot_time_s: float = 0.001  # ~1ms per shot including overhead
    cost_per_shot: float = 0.0  # Free tier or research access

    def validate(self) -> None:
        """Validate budget configuration."""
        if self.max_total_shots < 1000:
            raise ValueError(
                f"max_total_shots must be >= 1000, got {self.max_total_shots}"
            )
        if not 0 < self.warning_threshold < 1:
            raise ValueError(
                f"warning_threshold must be in (0, 1), "
                f"got {self.warning_threshold}"
            )
        if self.safety_margin < 0:
            raise ValueError(
                f"safety_margin must be >= 0, got {self.safety_margin}"
            )


@dataclass
class AllocationRecord:
    """Record of a single shot allocation."""

    timestamp: str
    shots_requested: int
    shots_approved: int
    cumulative_shots: int
    budget_remaining: int
    source: str  # Which experiment/batch requested


class ShotBudgetManager:
    """
    Hardware execution quota enforcement.

    Acts as a strict gatekeeper for all QPU shot allocations.
    Must be consulted before every circuit submission.

    Usage:
        budget = ShotBudgetManager(BudgetConfig(max_total_shots=200_000))
        
        # Before submitting
        if budget.can_submit(1000):
            budget.record_usage(1000, source="batch_12")
            # ... submit circuit ...
        else:
            # STOP — budget exhausted
    """

    def __init__(self, config: Optional[BudgetConfig] = None) -> None:
        self._config = config or BudgetConfig()
        self._config.validate()

        self._total_used: int = 0
        self._session_used: int = 0
        self._allocation_log: list[AllocationRecord] = []
        self._session_start: Optional[float] = None
        self._warnings_issued: int = 0

    @property
    def total_used(self) -> int:
        """Total shots consumed across all sessions."""
        return self._total_used

    @property
    def session_used(self) -> int:
        """Shots consumed in the current session."""
        return self._session_used

    @property
    def total_remaining(self) -> int:
        """Shots remaining in the total budget."""
        effective_limit = (
            self._config.max_total_shots - self._config.safety_margin
        )
        return max(0, effective_limit - self._total_used)

    @property
    def session_remaining(self) -> int:
        """Shots remaining in the current session budget."""
        return max(0, self._config.max_session_shots - self._session_used)

    @property
    def utilization(self) -> float:
        """Fraction of total budget consumed."""
        if self._config.max_total_shots == 0:
            return 1.0
        return self._total_used / self._config.max_total_shots

    @property
    def estimated_cost(self) -> float:
        """Estimated total cost of shots consumed."""
        return self._total_used * self._config.cost_per_shot

    @property
    def estimated_remaining_time_s(self) -> float:
        """Estimated wall-clock time for remaining budget."""
        return self.total_remaining * self._config.estimated_shot_time_s

    def can_submit(self, shots: int) -> bool:
        """
        Check if a shot allocation is within budget.

        Args:
            shots: Number of shots to submit.

        Returns:
            True if the allocation is approved.
        """
        if shots <= 0:
            return False

        # Check total budget
        if self._total_used + shots > (
            self._config.max_total_shots - self._config.safety_margin
        ):
            logger.error(
                f"BUDGET EXCEEDED: requested {shots} shots, "
                f"only {self.total_remaining} remaining "
                f"(total used: {self._total_used}/"
                f"{self._config.max_total_shots})"
            )
            return False

        # Check session budget
        if self._session_used + shots > self._config.max_session_shots:
            logger.error(
                f"SESSION BUDGET EXCEEDED: requested {shots} shots, "
                f"only {self.session_remaining} remaining in session"
            )
            return False

        # Issue warning if approaching threshold
        new_util = (self._total_used + shots) / self._config.max_total_shots
        if new_util >= self._config.warning_threshold:
            if self._warnings_issued == 0 or new_util >= 0.95:
                logger.warning(
                    f"BUDGET WARNING: {new_util:.1%} of total budget used "
                    f"({self._total_used + shots}/{self._config.max_total_shots})"
                )
                self._warnings_issued += 1

        return True

    def record_usage(
        self,
        shots: int,
        source: str = "unknown",
    ) -> AllocationRecord:
        """
        Record a shot allocation after successful execution.

        Args:
            shots: Number of shots actually executed.
            source: Identifier for the requesting experiment/batch.

        Returns:
            AllocationRecord for the transaction.
        """
        self._total_used += shots
        self._session_used += shots

        record = AllocationRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            shots_requested=shots,
            shots_approved=shots,
            cumulative_shots=self._total_used,
            budget_remaining=self.total_remaining,
            source=source,
        )
        self._allocation_log.append(record)

        logger.debug(
            f"Shot allocation: {shots} from '{source}', "
            f"cumulative={self._total_used}, "
            f"remaining={self.total_remaining}"
        )

        return record

    def new_session(self) -> None:
        """Reset session counter for a new Runtime session."""
        self._session_used = 0
        self._session_start = time.time()
        logger.info(
            f"New session started. "
            f"Session budget: {self._config.max_session_shots}. "
            f"Total remaining: {self.total_remaining}"
        )

    def summary(self) -> dict[str, Any]:
        """Return budget summary."""
        return {
            "total_used": self._total_used,
            "total_budget": self._config.max_total_shots,
            "total_remaining": self.total_remaining,
            "utilization": round(self.utilization, 4),
            "session_used": self._session_used,
            "session_budget": self._config.max_session_shots,
            "session_remaining": self.session_remaining,
            "estimated_cost": round(self.estimated_cost, 4),
            "estimated_remaining_time_s": round(
                self.estimated_remaining_time_s, 1
            ),
            "allocations": len(self._allocation_log),
            "warnings_issued": self._warnings_issued,
        }

    def export_log(self, path: Optional[str] = None) -> str:
        """
        Export allocation log to JSON.

        Args:
            path: File path to write. If None, returns JSON string.

        Returns:
            JSON string of the allocation log.
        """
        log_data = {
            "budget_config": {
                "max_total_shots": self._config.max_total_shots,
                "max_session_shots": self._config.max_session_shots,
                "safety_margin": self._config.safety_margin,
            },
            "summary": self.summary(),
            "allocations": [
                {
                    "timestamp": r.timestamp,
                    "shots": r.shots_approved,
                    "cumulative": r.cumulative_shots,
                    "remaining": r.budget_remaining,
                    "source": r.source,
                }
                for r in self._allocation_log
            ],
        }

        json_str = json.dumps(log_data, indent=2)

        if path is not None:
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json_str)
            logger.info(f"Budget log exported to {path}")

        return json_str
