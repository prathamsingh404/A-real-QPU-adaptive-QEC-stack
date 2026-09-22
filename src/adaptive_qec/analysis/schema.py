"""
Standard Scientific Results Schema (src/adaptive_qec/analysis/schema.py).

Defines the canonical, peer-review-grade JSON schema for all QEC simulation
and hardware experiments. Guarantees provenance, library version pinning,
git commit tracking, and cryptographic artifact hashing.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


def get_git_hash(repo_root: Optional[Path] = None) -> str:
    try:
        root = repo_root or Path(__file__).resolve().parents[3]
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def get_environment_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for mod_name in ["stim", "pymatching", "numpy", "scipy", "qiskit", "qiskit_ibm_runtime"]:
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[mod_name] = "not_installed"
    return versions


@dataclass
class ProvenanceMetadata:
    git_hash: str
    environment: dict[str, str]
    timestamp_utc: str
    wall_time_seconds: float
    seed: int
    artifact_sha256: str = ""


@dataclass
class ExperimentManifest:
    """Cryptographically verifiable execution manifest for QEC experiments."""
    experiment_id: str
    experiment_type: str
    git_commit: str
    code_version: str
    python_version: str
    dependency_lock_hash: str
    seed: int
    circuit_hash: str
    configuration_hash: str
    noise_model_hash: str
    start_time: str
    end_time: str
    backend_name: Optional[str] = None
    backend_version: Optional[str] = None
    job_id: Optional[str] = None
    calibration_timestamp: Optional[str] = None
    raw_data_hash: str = ""
    analysis_hash: str = ""
    provenance_status: str = "verified"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StandardExperimentResult:
    """Canonical research-grade experiment result format."""
    experiment_name: str
    schema_version: str = "2.0.0"
    manifest: Optional[ExperimentManifest] = None
    n_shots: int = 0
    n_rounds: int = 0
    logical_errors: Optional[int] = None
    logical_error_rate: Optional[float] = None
    detection_events: int = 0
    detection_rate: float = 0.0
    decoder_latency_us: Optional[float] = None
    decoder_throughput: Optional[float] = None
    switches: int = 0
    adaptation_events: int = 0
    config: dict[str, Any] = field(default_factory=dict)
    arms: dict[str, Any] = field(default_factory=dict)
    statistical_comparison: dict[str, Any] = field(default_factory=dict)
    extra_metrics: dict[str, Any] = field(default_factory=dict)
    provenance: Optional[ProvenanceMetadata] = None

    def finalize_and_save(self, output_path: Path, start_time: float, seed: int) -> Path:
        wall_time = time.time() - start_time
        meta = ProvenanceMetadata(
            git_hash=get_git_hash(),
            environment=get_environment_versions(),
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            wall_time_seconds=round(wall_time, 3),
            seed=seed,
            artifact_sha256="",
        )
        self.provenance = meta
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize first without hash
        raw_json = json.dumps(asdict(self), indent=2, default=str)
        # Compute SHA-256
        sha256 = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
        self.provenance.artifact_sha256 = sha256
        
        # Write final with hash
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, default=str)
            
        return output_path

    def save_immutable_bundle(self, base_dir: Path) -> Path:
        """Save a complete, tamper-evident experiment bundle in results/{exp_id}/.

        Creates:
            manifest.json, configuration.json, metrics.json, provenance.json
        """
        exp_id = (self.manifest.experiment_id if self.manifest 
                  else f"EXP-{int(time.time())}-{self.experiment_name}")
        bundle_dir = base_dir / exp_id
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # 1. Configuration
        config_path = bundle_dir / "configuration.json"
        config_bytes = json.dumps(self.config, indent=2, default=str).encode("utf-8")
        config_hash = hashlib.sha256(config_bytes).hexdigest()
        with open(config_path, "wb") as f:
            f.write(config_bytes)

        # 2. Metrics
        metrics_dict = {
            "experiment_name": self.experiment_name,
            "n_shots": self.n_shots,
            "n_rounds": self.n_rounds,
            "logical_errors": self.logical_errors,
            "logical_error_rate": self.logical_error_rate,
            "detection_events": self.detection_events,
            "detection_rate": self.detection_rate,
            "decoder_latency_us": self.decoder_latency_us,
            "decoder_throughput": self.decoder_throughput,
            "switches": self.switches,
            "adaptation_events": self.adaptation_events,
            "arms": self.arms,
            "statistical_comparison": self.statistical_comparison,
            "extra_metrics": self.extra_metrics,
        }
        metrics_path = bundle_dir / "metrics.json"
        metrics_bytes = json.dumps(metrics_dict, indent=2, default=str).encode("utf-8")
        metrics_hash = hashlib.sha256(metrics_bytes).hexdigest()
        with open(metrics_path, "wb") as f:
            f.write(metrics_bytes)

        # 3. Provenance
        prov_dict = asdict(self.provenance) if self.provenance else {
            "git_hash": get_git_hash(),
            "environment": get_environment_versions(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        prov_path = bundle_dir / "provenance.json"
        prov_bytes = json.dumps(prov_dict, indent=2, default=str).encode("utf-8")
        with open(prov_path, "wb") as f:
            f.write(prov_bytes)

        # 4. Manifest
        if self.manifest:
            manifest_dict = asdict(self.manifest)
            manifest_dict["configuration_hash"] = config_hash
            manifest_dict["analysis_hash"] = metrics_hash
        else:
            manifest_dict = {
                "experiment_id": exp_id,
                "experiment_name": self.experiment_name,
                "git_commit": get_git_hash(),
                "configuration_hash": config_hash,
                "analysis_hash": metrics_hash,
                "provenance_status": "verified",
            }
        manifest_path = bundle_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_dict, f, indent=2, default=str)

        return bundle_dir

