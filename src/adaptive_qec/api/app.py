"""
FastAPI application for AdaptiveQEC.

Provides REST API endpoints and web interface for:
    - QPU hardware telemetry & topology
    - Real QEC experiment generation & execution (Stim)
    - Decoder benchmarks (MWPM, Union-Find, ML Predecoder, Adaptive Router)
    - Noise statistics & CUSUM/EWMA drift detection
    - Adaptive control & selective recalibration
    - Hardware vs simulation gap analysis
    - Real-time systems profiling & latency budgeting
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from adaptive_qec.config import NoiseConfig
from adaptive_qec.data.store import ExperimentStore
from adaptive_qec.decoders.mwpm import MWPMDecoder
from adaptive_qec.decoders.union_find import UnionFindDecoder
from adaptive_qec.analysis.threshold import ThresholdAnalyzer
from adaptive_qec.experiment.distance_sweep import DistanceSweep
from adaptive_qec.noise.drift import CompositeDriftDetector, CUSUMDriftDetector, EWMADriftDetector
from adaptive_qec.noise.statistics import (
    compute_detector_statistics,
    compute_spatial_correlation,
    compute_temporal_correlation,
)
from adaptive_qec.qec.codes import create_code

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# API defaults (all from config or env — no magic numbers in endpoints)
# ---------------------------------------------------------------------------
_DEFAULT_BACKEND = os.environ.get("IBM_QUANTUM_BACKEND", "ibm_marrakesh")
_DEFAULT_PROVIDER = "IBM Quantum"
_DEFAULT_PROCESSOR_TYPE = "Heron r2"
_DEFAULT_NUM_QUBITS = 156
_DEFAULT_PHYSICAL_ERROR_RATE = 0.003021  # ECR gate error from calibration


app = FastAPI(
    title="AdaptiveQEC API",
    description="Hardware-aware, adaptive, low-latency quantum error correction",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---- Request / Response Models ----

class QECRunRequest(BaseModel):
    code_type: str = "surface"
    distance: int = 3
    rounds: int = 3
    basis: str = "Z"
    physical_error_rate: float = _DEFAULT_PHYSICAL_ERROR_RATE
    shots: int = 200


class BenchmarkRequest(BaseModel):
    distance: int = 3
    rounds: int = 3
    shots: int = 200
    physical_error_rate: float = _DEFAULT_PHYSICAL_ERROR_RATE


class ThresholdRequest(BaseModel):
    distances: list[int] = [3, 5]
    physical_error_rate: float = _DEFAULT_PHYSICAL_ERROR_RATE
    shots: int = 300
    code_type: str = "surface"


class RecalibrateRequest(BaseModel):
    target_error_rate: float = 0.015
    budget_shots: int = 2000


