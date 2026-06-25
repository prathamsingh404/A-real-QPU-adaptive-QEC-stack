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

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "idle_time_us": self.idle_time_us,
            "t2_us": self.t2_us,
            "block_order": self.block_order,
            "optimal_sequence": self.optimal_sequence,
            "pulse_damage_threshold_us": self.pulse_damage_threshold_us,
            "performances": {
                k: {
                    "sequence": v.sequence,
                    "pulse_count": v.pulse_count,
                    "pulse_overhead": v.pulse_overhead,
                    "suppression_factor": v.suppression_factor,
                    "effective_p_2q": v.effective_p_2q,
                    "logical_error_rate": v.logical_error_rate,
                    "detector_defect_rate": v.detector_defect_rate,
                    "samples_evaluated": v.samples_evaluated,
                }
                for k, v in self.performances.items()
            },
        }


def run_randomized_block_dd_benchmark(
    idle_time_us: float = 25.0,
    t2_us: float = 40.0,
    p_1q_pulse: float = 0.0003,
    nominal_p_2q: float = 0.003,
    shots_per_block: int = 500,
    seed: int = 2026,
) -> RandomizedBlockDDResult:
    """
    Run randomized block DD evaluation across Free Evolution, CPMG, XY4, XY8.
    """
    rng = np.random.default_rng(seed)
    sequences = [
        DDSequenceType.NONE,
        DDSequenceType.CPMG,
        DDSequenceType.XY4,
        DDSequenceType.XY8,
    ]

    seq_vals = [s.value for s in sequences]
    p1 = list(rng.permutation(seq_vals))
    p2 = list(rng.permutation(seq_vals))
    block_order = [str(x) for x in p1 + p2]

    PULSE_COUNTS = {
        DDSequenceType.NONE: 0,
        DDSequenceType.CPMG: 2,
        DDSequenceType.XY4: 4,
        DDSequenceType.XY8: 8,
    }
    SUPPRESSIONS = {
        DDSequenceType.NONE: 1.0,
        DDSequenceType.CPMG: 0.45,
        DDSequenceType.XY4: 0.15,
        DDSequenceType.XY8: 0.08,
    }

    # Free dephasing probability: p_phi = 0.5 * (1 - exp(-t_idle / T2))
    p_phi_free = 0.5 * (1.0 - np.exp(-idle_time_us / max(t2_us, 1e-6)))
