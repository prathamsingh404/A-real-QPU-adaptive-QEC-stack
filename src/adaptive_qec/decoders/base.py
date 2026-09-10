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
