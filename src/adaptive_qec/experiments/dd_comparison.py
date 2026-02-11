"""
Dynamical Decoupling (DD) Randomized Block Benchmark (TASK-052, TASK-054, TASK-056).

Compares Free Evolution (NONE), CPMG, XY4, and XY8 under randomized block designs (A/B/B/A).
Decouples syndrome detection rate R_det from logical fidelity and measures pulse damage
versus dephasing suppression as a function of idle window duration and T2 dephasing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import stim

from adaptive_qec.analysis.schema import StandardExperimentResult
from adaptive_qec.mitigation.dynamical_decoupling import DDSequenceType

logger = logging.getLogger(__name__)


@dataclass
class DDSequencePerformance:
    sequence: str
    pulse_count: int
    pulse_overhead: float
    suppression_factor: float
    effective_p_2q: float
    logical_error_rate: float
    detector_defect_rate: float
    samples_evaluated: int


@dataclass
class RandomizedBlockDDResult:
    experiment_id: str
    idle_time_us: float
    t2_us: float
    block_order: list[str]
    performances: dict[str, DDSequencePerformance]
    optimal_sequence: str
    pulse_damage_threshold_us: float
