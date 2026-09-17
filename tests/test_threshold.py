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


def _mock_metrics(shots: int, errors: int, latency_us: float = 5.0) -> DecoderMetrics:
    """Helper to create realistic DecoderMetrics for testing."""
    ler = errors / shots if shots > 0 else 0.0
    return DecoderMetrics(
        total_shots=shots,
        num_logical_errors=errors,
        logical_error_rate=ler,
        decode_time_s=0.01,
        per_shot_latency_us=np.full(shots, latency_us),
        latency_mean_us=latency_us,
        latency_p50_us=latency_us,
        latency_p95_us=latency_us,
        latency_p99_us=latency_us,
