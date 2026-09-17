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
        return {
            "total_wall_time_s": round(self.total_wall_time_s, 3),
            "noise_config": self.noise_config,
            "results": [
                {
                    "distance": r.distance,
                    "rounds": r.rounds,
                    "decoder": r.decoder_name,
                    "logical_error_rate": round(r.metrics.logical_error_rate, 8),
                    "num_errors": r.metrics.num_logical_errors,
                    "total_shots": r.metrics.total_shots,
                    "decode_time_s": round(r.metrics.decode_time_s, 4),
                    "throughput_shots_per_s": round(r.metrics.throughput_shots_per_s, 1),
                    "latency_p99_us": round(r.metrics.latency_p99_us, 2),
                    "physical_error_rate": round(r.physical_error_rate, 6),
                    "wall_time_s": round(r.wall_time_s, 3),
                }
                for r in self.results
            ],
        }


class DistanceSweep:
    """
    Run QEC experiments across multiple code distances.

    Generates surface code circuits, samples syndromes, decodes, and
    collects metrics for threshold scaling analysis.
    """

    def __init__(
        self,
        distances: list[int] | None = None,
        rounds_per_distance: dict[int, int] | None = None,
        noise: NoiseConfig | None = None,
        decoder_names: list[str] | None = None,
        shots_per_distance: int = 10000,
        code_type: str = "surface",
    ) -> None:
        """
        Args:
            distances: Code distances to sweep. Default: [3, 5, 7]
            rounds_per_distance: Rounds for each distance. Default: d rounds.
            noise: Noise configuration. Default: NoiseConfig with p=0.005.
            decoder_names: Decoders to benchmark. Default: ["mwpm"].
            shots_per_distance: Shots per (distance, decoder) pair.
            code_type: Code type ("surface" or "repetition").
        """
        self.distances = distances or [3, 5, 7]
        self.rounds_per_distance = rounds_per_distance or {}
        self.noise = noise or NoiseConfig()
        self.decoder_names = decoder_names or ["mwpm"]
        self.shots = shots_per_distance
        self.code_type = code_type

        # Default rounds = distance
        for d in self.distances:
            if d not in self.rounds_per_distance:
                self.rounds_per_distance[d] = d

    def run(self) -> DistanceSweepResults:
        """
        Execute the distance sweep.

        Returns:
            DistanceSweepResults with all (distance, decoder) results.
        """
        results = DistanceSweepResults(
            noise_config={
                "two_qubit": self.noise.gate.two_qubit,
            },
        )

        t_total_start = time.perf_counter()
        p_phys = self.noise.gate.two_qubit

        for d in sorted(self.distances):
            rounds = self.rounds_per_distance[d]
            logger.info(f"=== Distance {d}, {rounds} rounds, {self.shots} shots ===")

            # Generate circuit
            code = create_code(self.code_type, distance=d, rounds=rounds)
            circuit = code.generate_circuit(noise=self.noise)

            # Sample syndrome data (shared across decoders)
            sampler = circuit.compile_detector_sampler()
            detectors, observables = sampler.sample(
                shots=self.shots, separate_observables=True
            )

