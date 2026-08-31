"""
Abstract decoder interface.

All decoders are plug-ins:
    Decoder
    ├── lookup
    ├── MWPM
    ├── Union-Find
    ├── Belief Propagation
    ├── OSD
    ├── CNN
    ├── GNN
    ├── Transformer
    └── Adaptive decoder
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Correction:
    """Decoder output: predicted correction for each observable."""
    observable_corrections: np.ndarray  # shape (num_observables,) or (batch, num_observables)
    confidence: Optional[np.ndarray] = None  # P(correction | S) for each shot
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecoderMetrics:
    """Performance metrics for a decoder run."""
    total_shots: int
    num_logical_errors: int
    logical_error_rate: float
    decode_time_s: float
    per_shot_latency_us: Optional[np.ndarray] = None  # (shots,)
    latency_mean_us: float = 0.0
    latency_p50_us: float = 0.0
    latency_p95_us: float = 0.0
    latency_p99_us: float = 0.0
    latency_p999_us: float = 0.0
    throughput_shots_per_s: float = 0.0
    peak_memory_mb: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "total_shots": self.total_shots,
            "num_logical_errors": self.num_logical_errors,
            "logical_error_rate": self.logical_error_rate,
            "decode_time_s": self.decode_time_s,
            "latency_mean_us": self.latency_mean_us,
            "latency_p50_us": self.latency_p50_us,
            "latency_p95_us": self.latency_p95_us,
            "latency_p99_us": self.latency_p99_us,
            "latency_p999_us": self.latency_p999_us,
            "throughput_shots_per_s": self.throughput_shots_per_s,
            "peak_memory_mb": self.peak_memory_mb,
            "extra": self.extra,
        }


class Decoder(ABC):
    """
    Abstract decoder interface.

    Every decoder must implement decode() and return both corrections
    and comprehensive performance metrics.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Decoder name for logging and storage."""
        ...

    @abstractmethod
    def configure(self, **kwargs: Any) -> None:
        """
        Configure the decoder with code-specific parameters.

        Called once before decoding, e.g., with a DetectorErrorModel.
        """
        ...

    @abstractmethod
    def decode(self, syndrome: np.ndarray) -> Correction:
        """
        Decode a syndrome array.

        Args:
            syndrome: shape (num_detectors,) for single shot, or
                      (batch, num_detectors) for batch decoding.

        Returns:
            Correction with predicted observable corrections.
        """
        ...

    @abstractmethod
    def decode_batch(
        self,
        syndromes: np.ndarray,
        observable_flips: np.ndarray,
    ) -> DecoderMetrics:
        """
        Decode a batch of syndromes and compute performance metrics.

        Args:
            syndromes: shape (shots, num_detectors)
            observable_flips: shape (shots, num_observables) — ground truth

        Returns:
            DecoderMetrics with error rate, latency, throughput.
        """
        ...
