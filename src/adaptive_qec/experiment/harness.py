"""
Experiment harness: the unified runner for adaptive-vs-static experiments.

Orchestrates the closed loop:
    hardware → syndrome extraction → decoding → reward → controller update

Supports:
    - A/B comparison (static baseline vs. adaptive controller)
    - Multi-arm comparison (multiple controllers on same hardware stream)
    - Interleaved execution (round-robin across controllers)
    - Telemetry and provenance logging

The harness uses the QPU backend (IBMBackend or mock) and decoder
interfaces directly — no simulation shortcuts.

Usage:
    harness = ExperimentHarness.from_config("configs/default.yaml")
    results = harness.run(num_windows=100)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import stim

from adaptive_qec.config import AQECConfig, load_config
from adaptive_qec.controller.base import BaseController, TelemetryRecord
from adaptive_qec.controller.controller import (
    ControlAction,
    DecoderChoice,
    HardwareState,
)
from adaptive_qec.decoders.base import Decoder, DecoderMetrics
from adaptive_qec.decoders.mwpm import MWPMDecoder
from adaptive_qec.noise.drift import DriftStatus, EWMADriftDetector
from adaptive_qec.provenance import DataProvenance, ProvenanceRegistry, ProvenanceTag
from adaptive_qec.syndrome.extraction import SyndromeExtractor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment result
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    """Result from a single observation window."""
    window_index: int
    controller_name: str
    action: dict[str, Any]
    hardware_state: dict[str, Any]
    decoder_metrics: dict[str, Any]
    logical_error_rate: float
    reward: float  # 1 - logical_error_rate
    elapsed_ms: float
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_index": self.window_index,
            "controller_name": self.controller_name,
            "action": self.action,
            "hardware_state": self.hardware_state,
            "decoder_metrics": self.decoder_metrics,
            "logical_error_rate": self.logical_error_rate,
            "reward": self.reward,
            "elapsed_ms": self.elapsed_ms,
            "timestamp": self.timestamp,
        }


@dataclass
class ExperimentRunResult:
    """Complete result from an experiment run."""
    experiment_id: str
    config: dict[str, Any]
    controller_summaries: dict[str, Any]
    window_results: list[dict[str, Any]]
    aggregate_metrics: dict[str, Any]
    provenance: dict[str, Any]
    start_time: str
    end_time: str
    total_windows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "config": self.config,
            "controller_summaries": self.controller_summaries,
            "window_results": self.window_results,
            "aggregate_metrics": self.aggregate_metrics,
            "provenance": self.provenance,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_windows": self.total_windows,
        }


# ---------------------------------------------------------------------------
# Stim circuit builder (uses real noise parameters, not hardcoded)
# ---------------------------------------------------------------------------

def build_surface_code_circuit(
    distance: int,
    rounds: int,
    p_1q: float,
    p_2q: float,
    p_ro: float,
) -> stim.Circuit:
    """Build a rotated surface-code memory circuit with calibrated noise.

    Noise rates come from hardware calibration data, not constants.

    Parameters
    ----------
