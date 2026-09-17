"""
QEC code definitions.

Defines quantum error-correcting codes with their Stim circuit representations.
Supports repetition code and rotated surface code with parameterizable
distance and rounds.

The generated Stim circuits include:
- Full stabilizer measurement circuits
- Detector annotations
- Observable annotations
- Configurable noise injection points
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import stim

from adaptive_qec.config import NoiseConfig

logger = logging.getLogger(__name__)


@dataclass
class CodeInfo:
    """Static information about a QEC code instance."""
    name: str
    distance: int
