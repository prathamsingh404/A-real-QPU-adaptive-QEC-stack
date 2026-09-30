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
