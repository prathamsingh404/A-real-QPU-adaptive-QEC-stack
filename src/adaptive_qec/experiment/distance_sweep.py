"""
Automated distance sweep experiment runner.

Runs QEC experiments across multiple code distances to collect the data
needed for threshold scaling analysis (Lambda ratio computation).

For each distance d:
    1. Generate a surface code circuit with the specified noise model
    2. Sample N shots from the Stim sampler
    3. Decode using the specified decoder(s)
    4. Collect DecoderMetrics

The results feed into ThresholdAnalyzer for Lambda computation.

Usage:
    sweep = DistanceSweep(
        distances=[3, 5, 7],
        rounds_per_distance=None,  # defaults to d
        noise=NoiseConfig(gate=GateNoiseConfig(two_qubit=0.005)),
        decoder_names=["mwpm", "union_find"],
        shots_per_distance=10000,
    )
    results = sweep.run()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from adaptive_qec.config import NoiseConfig
from adaptive_qec.decoders.base import DecoderMetrics
from adaptive_qec.decoders.registry import get_decoder
from adaptive_qec.qec.codes import create_code

logger = logging.getLogger(__name__)


@dataclass
class SweepResult:
    """Result for a single (distance, decoder) combination."""
    distance: int
    rounds: int
    decoder_name: str
    metrics: DecoderMetrics
    physical_error_rate: float
    wall_time_s: float


@dataclass
class DistanceSweepResults:
    """Complete results from a distance sweep experiment."""
    results: list[SweepResult] = field(default_factory=list)
    total_wall_time_s: float = 0.0
    noise_config: Optional[dict[str, Any]] = None

    def get_by_decoder(self, decoder_name: str) -> list[SweepResult]:
        """Get results for a specific decoder."""
        return [r for r in self.results if r.decoder_name == decoder_name]

    def get_by_distance(self, distance: int) -> list[SweepResult]:
        """Get results for a specific distance."""
        return [r for r in self.results if r.distance == distance]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
