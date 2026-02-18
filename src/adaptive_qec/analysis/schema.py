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
