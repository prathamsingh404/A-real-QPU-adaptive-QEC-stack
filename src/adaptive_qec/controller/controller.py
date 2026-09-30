"""
Physics-Gated Adaptive QEC Controller
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class DecoderChoice(Enum):
    MWPM = 'mwpm'
    UNION_FIND = 'union_find'
    LAZY_MWPM = 'lazy_mwpm'

class DDPattern(Enum):
    NONE = 'none'
    CPMG = 'cpmg'
    XY4 = 'xy4'
    EDD = 'edd'

@dataclass
class HardwareTelemetry:
    drift_magnitude: float = 0.0
    burst_detected: bool = False
    leakage_fraction: float = 0.0
    qubit_defect_rate: float = 0.0
    code_distance: int = 3
    round_index: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass
class ControllerAction:
    decoder: DecoderChoice
    dd_pattern: DDPattern
    recalibrate_dem: bool = False
    reason: str = 'nominal'

class AdaptiveController:
    """Physics-Gated Adaptive Controller for Real-Time QEC Runtime."""
    def __init__(self, hysteresis_margin: float = 0.05, leakage_threshold: float = 0.15, distance_crossover: int = 5) -> None:
        self.hysteresis_margin = hysteresis_margin
        self.leakage_threshold = leakage_threshold
        self.distance_crossover = distance_crossover
        self.current_action = ControllerAction(decoder=DecoderChoice.MWPM, dd_pattern=DDPattern.NONE, reason='initial')
        self.fast_path_count = 0
