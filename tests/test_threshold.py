"""Tests for Threshold Scaling Analysis (Problem 5).

Validates Lambda ratio computation (Λ), phenomenological threshold fitting,
and automated distance sweep experiments.
"""

import numpy as np
import pytest

from adaptive_qec.analysis.threshold import ThresholdAnalyzer, ThresholdFit
from adaptive_qec.config import NoiseConfig
from adaptive_qec.decoders.base import DecoderMetrics
from adaptive_qec.experiment.distance_sweep import DistanceSweep, DistanceSweepResults

