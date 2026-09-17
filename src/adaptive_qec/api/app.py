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


# ---- Global State / Cache ----

from adaptive_qec.noise.drift import CompositeDriftDetector, DriftStatus

_CURRENT_EXPERIMENT: dict[str, Any] = {}
_DRIFT_DETECTOR = CompositeDriftDetector(
    ewma_alpha=0.15,
    cusum_threshold=4.5,
    warmup_samples=5,
)
_LATEST_DRIFT_REPORT = None
_DRIFT_HISTORY: list[dict[str, Any]] = []
_EXPERIMENT_HISTORY: list[dict[str, Any]] = []


# ---- Endpoints ----

@app.get("/")
async def root():
    """Serve the ethereal frontend dashboard."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "service": "AdaptiveQEC",
        "version": "0.1.0",
        "status": "operational",
    }


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/qpu/telemetry")
async def get_qpu_telemetry() -> dict[str, Any]:
    """
    Get QPU hardware telemetry.

    Returns data from the last real experiment if available,
    otherwise returns baseline data sourced from IBM Quantum
    calibration for the configured backend.
    """
    # Check if we have data from a real experiment run
    last_exp = _CURRENT_EXPERIMENT
    has_real_data = bool(last_exp.get("detector_rates"))

    backend_name = last_exp.get("backend", _DEFAULT_BACKEND)
    num_qubits = _DEFAULT_NUM_QUBITS

    # If we have real experiment data, derive telemetry from it
    if has_real_data:
        mean_defect_rate = last_exp.get("mean_defect_rate", 0.0)
        detector_rates = last_exp.get("detector_rates", [])
        source = "measured"
    else:
        mean_defect_rate = 0.0
        detector_rates = []
        source = "baseline"

    # Build topology from backend configuration
    # For Heron r2 (ibm_marrakesh), use a representative subset
    # of the heavy-hex lattice for visualization
    vis_qubits = min(num_qubits, 27)  # Visualize a 27-qubit patch
    rng = np.random.default_rng(42)

    # Heavy-hex patch edges for visualization
    edges = [
