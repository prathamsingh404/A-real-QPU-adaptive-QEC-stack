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
