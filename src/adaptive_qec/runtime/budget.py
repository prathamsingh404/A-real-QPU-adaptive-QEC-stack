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
