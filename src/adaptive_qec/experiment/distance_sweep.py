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
