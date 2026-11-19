"""Research results, recorded device data and interactive Stim simulations."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from adaptive_qec.config import NoiseConfig
from adaptive_qec.data.store import ExperimentStore
from adaptive_qec.decoders.mwpm import MWPMDecoder
from adaptive_qec.decoders.union_find import UnionFindDecoder
from adaptive_qec.experiment.distance_sweep import DistanceSweep
from adaptive_qec.noise.statistics import (
    compute_detector_statistics,
    compute_spatial_correlation,
)
from adaptive_qec.qec.codes import create_code

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# API defaults (all from config or env — no magic numbers in endpoints)
# ---------------------------------------------------------------------------
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
    distance: int = Field(default=3, ge=2, le=31)
    rounds: int = Field(default=3, ge=1, le=64)
    basis: Literal["Z", "X"] = "Z"
    physical_error_rate: float = Field(default=_DEFAULT_PHYSICAL_ERROR_RATE, ge=0, le=1)
    shots: int = Field(default=200, ge=1, le=100000)


class BenchmarkRequest(BaseModel):
    distance: int = Field(default=3, ge=2, le=31)
    rounds: int = Field(default=3, ge=1, le=64)
    shots: int = Field(default=200, ge=1, le=100000)
    physical_error_rate: float = Field(default=_DEFAULT_PHYSICAL_ERROR_RATE, ge=0, le=1)


class ThresholdRequest(BaseModel):
    distances: list[int] = [3, 5]
    physical_error_rate: float = Field(default=_DEFAULT_PHYSICAL_ERROR_RATE, ge=0, le=1)
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
    snapshot = await get_live_calibration_snapshot()
    return {"source": "stored calibration snapshot", "metadata": snapshot["metadata"],
            "summary_statistics": snapshot["summary_statistics"]}


@app.post("/api/qec/run")
async def run_qec_experiment(req: QECRunRequest) -> dict[str, Any]:
    """
    Execute real QEC experiment on Stim circuit, sample detection events,
    extract syndrome tensor S in {0, 1}^(R x N_d), and decode with PyMatching MWPM.
    """
    try:
        code = create_code(
            code_type=req.code_type,
            distance=req.distance,
            rounds=req.rounds,
        )
        noise = NoiseConfig()
        noise.gate.two_qubit = req.physical_error_rate
        noise.gate.single_qubit = req.physical_error_rate * 0.15
        noise.readout.p0_given_1 = 0.01208
        noise.readout.p1_given_0 = 0.01208

        circuit = code.generate_circuit(noise=noise)
        if req.basis == "X":
            if req.code_type != "surface":
                raise HTTPException(422, "The X-memory option requires a surface code.")
            import stim
            circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=req.distance, rounds=req.rounds,
                                             after_clifford_depolarization=req.physical_error_rate,
                                             before_round_data_depolarization=req.physical_error_rate,
                                             before_measure_flip_probability=req.physical_error_rate,
                                             after_reset_flip_probability=req.physical_error_rate)
        sampler = circuit.compile_detector_sampler()

        t_start = time.perf_counter()
        detectors, observables = sampler.sample(shots=req.shots, separate_observables=True)
        t_sample = time.perf_counter() - t_start

        # MWPM decoding
        decoder = MWPMDecoder()
        decoder.configure(circuit=circuit)
        metrics = decoder.decode_batch(detectors, observables)

        # Compute syndrome defect rates
        num_detectors = detectors.shape[1]
        mean_defect_rate = float(np.mean(detectors))
        detector_rates = np.mean(detectors, axis=0).round(4).tolist()

        # Wilson score confidence interval for logical error rate
        n = req.shots
        p = metrics.logical_error_rate
        z = 1.96  # 95% CI
        denom = 1.0 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        delta = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        ci_lower = max(0.0, float(center - delta))
        ci_upper = min(1.0, float(center + delta))

        # Format sample matrix for interactive visualization
        sample_matrix = detectors[:min(12, req.shots)].astype(int).tolist()

        exp_data = {
            "source": "stim simulation",
            "experiment_id": f"qec_run_{int(time.time())}",
            "backend": "Stim",
            "code_type": req.code_type,
            "distance": req.distance,
            "rounds": req.rounds,
            "basis": req.basis,
            "shots": req.shots,
            "num_detectors": num_detectors,
            "num_observables": observables.shape[1],
            "logical_error_rate": round(p, 4),
            "logical_errors_count": metrics.num_logical_errors,
            "confidence_interval_95": [round(ci_lower, 4), round(ci_upper, 4)],
            "mean_defect_rate": round(mean_defect_rate, 4),
            "detector_rates": detector_rates[:30],
            "latency_mean_us": round(metrics.latency_mean_us, 2),
            "latency_p99_us": round(metrics.latency_p99_us, 2),
            "throughput_shots_per_s": round(metrics.throughput_shots_per_s, 0),
            "sample_matrix": sample_matrix,
            "detection_events": detectors,
        }

        # Update drift detector with real data
        global _LATEST_DRIFT_REPORT
        _LATEST_DRIFT_REPORT = _DRIFT_DETECTOR.update(np.array(detector_rates))

        # Record drift history
        _DRIFT_HISTORY.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "defect_rate": round(mean_defect_rate, 4),
            "drift_status": _LATEST_DRIFT_REPORT.status.name if _LATEST_DRIFT_REPORT else "UNKNOWN",
        })
        # Keep last 50 entries
        if len(_DRIFT_HISTORY) > 50:
            _DRIFT_HISTORY[:] = _DRIFT_HISTORY[-50:]

        # Store for other endpoints to reference
        _CURRENT_EXPERIMENT.update({k: v for k, v in exp_data.items() if k != "detection_events"})
        _CURRENT_EXPERIMENT["detectors_array"] = detectors

        # Record in history
        _EXPERIMENT_HISTORY.append({
            "experiment_id": exp_data["experiment_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logical_error_rate": exp_data["logical_error_rate"],
            "mean_defect_rate": exp_data["mean_defect_rate"],
            "distance": req.distance,
            "shots": req.shots,
        })
        if len(_EXPERIMENT_HISTORY) > 100:
            _EXPERIMENT_HISTORY[:] = _EXPERIMENT_HISTORY[-100:]

        return {k: v for k, v in exp_data.items() if k != "detection_events"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error executing QEC experiment")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/decoders/benchmark")
async def benchmark_decoders(req: BenchmarkRequest) -> dict[str, Any]:
    """
    Compare decoders side-by-side on identical syndrome data.

    MWPM (PyMatching) and Union-Find (Delfosse & Nickerson) run real execution.
    Both implementations score the same simulated observations.
    """
    code = create_code(
        code_type="surface",
        distance=req.distance,
        rounds=req.rounds,
    )
    noise = NoiseConfig()
    noise.gate.two_qubit = req.physical_error_rate
    circuit = code.generate_circuit(noise=noise)
    sampler = circuit.compile_detector_sampler()
    detectors, observables = sampler.sample(shots=req.shots, separate_observables=True)

    # 1. MWPM (PyMatching) — real execution
    mwpm = MWPMDecoder()
    mwpm.configure(circuit=circuit)
    m_metrics = mwpm.decode_batch(detectors, observables)

    # 2. Union-Find (Delfosse & Nickerson) — real execution
    uf = UnionFindDecoder()
    uf.configure(circuit=circuit)
    uf_metrics = uf.decode_batch(detectors, observables)

    return {"source":"stim simulation", "distance":req.distance, "rounds":req.rounds, "shots":req.shots,
            "decoders":[{"name":name, "logical_errors":m.num_logical_errors, "logical_error_rate":m.logical_error_rate,
                         "latency_mean_us":m.latency_mean_us, "latency_p99_us":m.latency_p99_us}
                        for name,m in [("PyMatching MWPM",m_metrics), ("Union-find heuristic",uf_metrics)]]}


@app.post("/api/analysis/threshold")
async def run_threshold_analysis(req: ThresholdRequest) -> dict[str, Any]:
    """
    Run distance sweep across requested code distances and compute
    the fault-tolerant threshold scaling metrics (Lambda ratio Λ and p_th).
    """
    try:
        noise = NoiseConfig()
        noise.gate.two_qubit = req.physical_error_rate
        noise.gate.single_qubit = req.physical_error_rate * 0.15

        sweep = DistanceSweep(
            distances=req.distances,
            noise=noise,
            decoder_names=["mwpm", "union_find"],
            shots_per_distance=req.shots,
            code_type=req.code_type,
        )
        sweep_results = sweep.run()

        return {"source":"stim simulation", "physical_error_rate":req.physical_error_rate,
                "scope":"Single physical error rate; threshold unestimated.", "sweep_summary":sweep_results.to_dict()}
    except Exception as e:
        logger.exception("Error running threshold analysis")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/noise/characterization")
async def get_noise_characterization() -> dict[str, Any]:
    """
    Hardware noise characterization from real experiment data:
        - Single-detector defect probabilities P(D_i = 1)
        - Pair correlation matrix C_ij = E[D_i D_j] - E[D_i]E[D_j]
        - Temporal correlation C(k) = corr(D_t, D_{t+k})
        - Real-time CUSUM / EWMA drift status
    """
    detectors = _CURRENT_EXPERIMENT.get("detectors_array")
    source = "stim simulation"
    if detectors is None:
        return {"source":"no_data", "message":"Run a simulated memory first."}

    stats = compute_detector_statistics(detectors)
    spatial_corr = compute_spatial_correlation(detectors)

    # Use real drift history from experiments
    global _LATEST_DRIFT_REPORT
    is_alarm = (_LATEST_DRIFT_REPORT is not None and
                _LATEST_DRIFT_REPORT.status in (DriftStatus.DRIFT_DETECTED, DriftStatus.SEVERE))
    magnitude = float(_LATEST_DRIFT_REPORT.magnitude) if _LATEST_DRIFT_REPORT else 0.0
    affected_qubits = _LATEST_DRIFT_REPORT.affected_detectors if _LATEST_DRIFT_REPORT else []

    return {
        "source": source,
        "num_detectors": int(stats.num_detectors),
        "mean_detection_rate": round(float(stats.mean_detection_rate), 4),
        "max_detection_rate": round(float(stats.max_detection_rate), 4),
        "hotspot_detectors": [int(h) for h in stats.hotspot_detectors],
        "temporal_lag_correlations": None,  # Requires round and spatial detector alignment.
        "significant_correlated_pairs": [[int(i), int(j), round(float(v), 4)] for i, j, v in spatial_corr.significant_pairs[:6]],
        "drift_status": {
            "status": "DRIFT DETECTED" if is_alarm else "STABLE",
            "is_drift": bool(is_alarm),
            "magnitude": round(float(magnitude), 3),
            "affected_detectors": [int(q) for q in affected_qubits],
            "history": _DRIFT_HISTORY[-10:] if _DRIFT_HISTORY else [],
        }
    }


@app.get("/api/adaptive/policy")
async def get_adaptive_policy() -> dict[str, Any]:
    import yaml
    return yaml.safe_load((ROOT / "configs/suite.yaml").read_text())


@app.post("/api/adaptive/calibrate")
async def run_selective_calibration(req: RecalibrateRequest):
    raise HTTPException(409, "Device recalibration requires an instrumented hardware run.")


@app.get("/api/simulator/gap")
async def get_simulator_gap():
    return {"source":"unavailable", "reason":"No paired physical and simulated runs with matched parameters."}


@app.get("/api/profiling/latency")
async def get_profiling_latency():
    return json.loads((ROOT / "data/host_timings.json").read_text())


@app.get("/api/study")
async def study():
    return json.loads((ROOT / "data/analysis.json").read_text())


@app.get("/api/study/replay")
async def replay_results():
    return json.loads((ROOT / "data/public_replay_results.json").read_text())


@app.get("/api/study/follow-up")
async def follow_up():
    return json.loads((ROOT / "data/timing_confirmation.json").read_text())


@app.get("/api/study/public-decoders")
async def public_decoders():
    return json.loads((ROOT / "data/public_decoder_comparison.json").read_text())


@app.get("/api/study/paper")
async def paper():
    return FileResponse(ROOT / "papers/main.pdf", media_type="application/pdf")


# ---- Experiment Store endpoints ----

@app.get("/api/experiments")
async def list_experiments(base_path: str = "experiments") -> dict[str, Any]:
    """List stored experiments."""
    store = ExperimentStore(base_path)
    experiments = store.list_experiments()
    return {"experiments": experiments, "count": len(experiments)}


@app.get("/api/experiments/history")
async def get_experiment_history() -> dict[str, Any]:
    """Get the history of experiments run in this session."""
    return {
        "history": _EXPERIMENT_HISTORY,
        "count": len(_EXPERIMENT_HISTORY),
    }


# ---- Real Hardware Data & Practical Benchmark Endpoints (Zero Mock Data) ----

@app.get("/api/hardware/practical-results")
async def get_practical_hardware_results() -> dict[str, Any]:
    """Return real empirical results from IBM Heron practical benchmarks (IQPE & Teleportation)."""
    path = ROOT / Path("data/hardware_results/ibm_marrakesh_practical_benchmarks_results.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Practical hardware results file not found.")
    with open(path) as f:
        return json.load(f)


@app.get("/api/hardware/quantum-results")
async def get_true_quantum_hardware_results() -> dict[str, Any]:
    """Return real empirical results from IBM Heron [[4, 2, 2]] code and dynamic feedforward."""
    path = ROOT / Path("data/hardware_results/ibm_marrakesh_true_quantum_and_dynamic_results.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Quantum hardware results file not found.")
    with open(path) as f:
        return json.load(f)


@app.get("/api/hardware/qec-results")
async def get_qec_hardware_results() -> dict[str, Any]:
    """Return real empirical results from IBM Heron repetition code QEC experiment."""
    path = ROOT / Path("data/hardware_results/ibm_marrakesh_qec_results.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="QEC hardware results file not found.")
    with open(path) as f:
        return json.load(f)


@app.get("/api/hardware/calibration-live")
async def get_live_calibration_snapshot() -> dict[str, Any]:
    """Return the complete 156-qubit physical calibration snapshot from ibm_marrakesh."""
    path = ROOT / Path("data/calibration/ibm_marrakesh_live_snapshot.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Live calibration snapshot not found.")
    with open(path) as f:
        return json.load(f)


@app.get("/api/hardware/high-stats-50k")
async def get_high_stats_validation() -> dict[str, Any]:
    return {"status":"historical simulation", "current_results":"/api/study"}


class NoiseComparisonRequest(BenchmarkRequest):
    comparison_error_rate: float = Field(ge=0, le=1)


@app.post("/api/hardware/run-live-benchmark")
async def old_live_benchmark():
    raise HTTPException(410, "The former endpoint used assumed noise suppression. Use /api/simulator/noise-comparison for explicit simulation.")


@app.post("/api/simulator/noise-comparison")
async def noise_comparison(req: NoiseComparisonRequest):
    import stim
    from scipy.stats import fisher_exact

    runs = []
    for p in (req.physical_error_rate, req.comparison_error_rate):
        circuit = stim.Circuit.generated("repetition_code:memory", distance=req.distance, rounds=req.rounds,
                                         after_clifford_depolarization=p)
        x, y = circuit.compile_detector_sampler().sample(shots=req.shots, separate_observables=True)
        decoder = MWPMDecoder()
        decoder.configure(circuit=circuit)
        errors = decoder.decode_batch(x,y).num_logical_errors
        runs.append(dict(physical_error_rate=p, shots=req.shots, errors=errors, ler=errors/req.shots))
    contingency = [[r["errors"], req.shots-r["errors"]] for r in runs]
    return {"source":"two independent Stim simulations", "runs":runs,
            "two_sided_fisher_p":float(fisher_exact(contingency).pvalue)}
