"""Tests for Leakage Detection and Reset Protocol (Problem 6).

Validates temporal autocorrelation, streak persistence analysis,
leakage candidate detection, rate estimation (γ_L, γ_S), and digital twin
leakage tracking.
"""

import numpy as np
import pytest

from adaptive_qec.digital_twin.twin import HardwareDigitalTwin
from adaptive_qec.noise.characterization import NoiseCharacterizer
from adaptive_qec.noise.leakage import (
    LeakageAnalysis,
    LeakageDetector,
    LeakageRateEstimator,
