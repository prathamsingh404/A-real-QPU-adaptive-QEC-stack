"""
Reproducibility utilities.

Every experiment captures:
    - Git commit hash
    - Software versions (Python, stim, pymatching, qiskit, ...)
    - Full config snapshot
    - Timestamp

Six months later: `reproduce experiment_0421` should work.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, Union

logger = logging.getLogger(__name__)


class ManifestVerificationError(Exception):
    """Raised when cryptographic verification of manifest, config files, or seeds fails."""
    pass


@dataclass
class CryptographicManifest:
    """
    Cryptographic experiment provenance manifest verifying:
      - SHA-256 digests of all configuration files.
      - SHA-256 digest of random seed sequences.
      - Git commit and working tree cleanliness.
      - Exact software library versions.
    """
    manifest_version: str
    experiment_id: str
    timestamp_utc: str
    git_commit: Optional[str]
    git_status: Optional[str]
    software_versions: dict[str, str]
    config_hashes: dict[str, str]       # Path -> SHA-256 hex
    seeds: list[int]
    seeds_sha256: str                   # SHA-256 hex digest of serialized seeds
    manifest_digest: str                # SHA-256 digest binding all manifest components
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "experiment_id": self.experiment_id,
            "timestamp_utc": self.timestamp_utc,
            "git_commit": self.git_commit,
            "git_status": self.git_status,
            "software_versions": self.software_versions,
            "config_hashes": self.config_hashes,
            "seeds": self.seeds,
            "seeds_sha256": self.seeds_sha256,
            "manifest_digest": self.manifest_digest,
            "metadata": self.metadata,
        }


def hash_bytes_sha256(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_file_sha256(path: Union[str, Path]) -> str:
    """Compute SHA-256 hex digest of a file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found for hashing: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def hash_seeds(seeds: Union[int, Sequence[int]]) -> str:
    """Compute deterministic SHA-256 hex digest of a seed or sequence of seeds."""
    if isinstance(seeds, int):
        seed_list = [seeds]
    else:
        seed_list = [int(s) for s in seeds]
    canonical_repr = json.dumps(seed_list, separators=(",", ":"))
    return hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()


def generate_cryptographic_manifest(
    experiment_id: str,
    config_paths: Sequence[Union[str, Path]],
    seeds: Union[int, Sequence[int]],
    metadata: Optional[dict[str, Any]] = None,
) -> CryptographicManifest:
    """
    Generate a cryptographic manifest with SHA-256 digests of all config files and seeds.
    """
    if isinstance(seeds, int):
        seed_list = [seeds]
    else:
        seed_list = [int(s) for s in seeds]

    config_hashes: dict[str, str] = {}
    for cp in config_paths:
        p = Path(cp)
        norm_key = str(p).replace("\\", "/")
        config_hashes[norm_key] = hash_file_sha256(p)

    seeds_hash = hash_seeds(seed_list)
    repro = capture_reproducibility_info()
    timestamp = datetime.now(timezone.utc).isoformat()

    # Compute binding digest across git commit, configs, and seeds
    digest_payload = {
        "manifest_version": "2.0.0",
        "experiment_id": experiment_id,
        "git_commit": repro.get("git_commit"),
        "config_hashes": config_hashes,
        "seeds_sha256": seeds_hash,
    }
    canonical_payload = json.dumps(digest_payload, sort_keys=True, separators=(",", ":"))
    manifest_digest = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()

    return CryptographicManifest(
        manifest_version="2.0.0",
        experiment_id=experiment_id,
        timestamp_utc=timestamp,
        git_commit=repro.get("git_commit"),
        git_status=repro.get("git_status"),
        software_versions=repro.get("software_versions", {}),
        config_hashes=config_hashes,
        seeds=seed_list,
        seeds_sha256=seeds_hash,
        manifest_digest=manifest_digest,
        metadata=metadata or {},
    )


def verify_cryptographic_manifest(
    manifest: Union[CryptographicManifest, dict[str, Any]],
    config_base_dir: Optional[Union[str, Path]] = None,
    expected_seeds: Optional[Sequence[int]] = None,
    raise_on_error: bool = False,
) -> tuple[bool, list[str]]:
    """
    Verify the cryptographic integrity of a manifest:
      1. Verifies SHA-256 of all config files listed in config_hashes.
      2. Verifies SHA-256 of seeds against seeds_sha256.
      3. Verifies seeds against expected_seeds if provided.
      4. Verifies manifest_digest integrity.

    Returns (is_valid, error_messages).
    """
    m_dict = manifest.to_dict() if isinstance(manifest, CryptographicManifest) else manifest
    errors: list[str] = []

    # 1. Verify seeds hash
    seeds = m_dict.get("seeds", [])
    computed_seeds_hash = hash_seeds(seeds)
    expected_seeds_hash = m_dict.get("seeds_sha256", "")
    if computed_seeds_hash != expected_seeds_hash:
        errors.append(
            f"Seeds SHA-256 mismatch: recorded {expected_seeds_hash}, computed {computed_seeds_hash}"
        )

    # 2. Check against expected_seeds if provided
    if expected_seeds is not None:
        exp_list = [int(s) for s in expected_seeds]
        if list(seeds) != exp_list:
            errors.append(
                f"Seeds sequence mismatch: expected {exp_list}, found in manifest {list(seeds)}"
            )

    # 3. Verify config file hashes
    config_hashes = m_dict.get("config_hashes", {})
    base_dir = Path(config_base_dir) if config_base_dir else Path.cwd()
    for conf_path_str, expected_hash in config_hashes.items():
        conf_path = Path(conf_path_str)
        # Try direct path, then relative to base_dir
        target_path = conf_path if conf_path.is_file() else (base_dir / conf_path)
        if not target_path.is_file():
            errors.append(f"Config file not found on disk for verification: {conf_path_str}")
            continue

        try:
            actual_hash = hash_file_sha256(target_path)
            if actual_hash != expected_hash:
                errors.append(
                    f"Config hash mismatch for {conf_path_str}: expected {expected_hash}, got {actual_hash}"
                )
        except Exception as e:
            errors.append(f"Failed to read/hash config {conf_path_str}: {e}")

    # 4. Verify manifest binding digest
    digest_payload = {
        "manifest_version": m_dict.get("manifest_version", "2.0.0"),
        "experiment_id": m_dict.get("experiment_id", ""),
        "git_commit": m_dict.get("git_commit"),
        "config_hashes": config_hashes,
        "seeds_sha256": expected_seeds_hash,
    }
    canonical_payload = json.dumps(digest_payload, sort_keys=True, separators=(",", ":"))
    recomputed_digest = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
    if recomputed_digest != m_dict.get("manifest_digest"):
        errors.append(
            f"Manifest binding digest mismatch: recorded {m_dict.get('manifest_digest')}, computed {recomputed_digest}"
        )

    is_valid = len(errors) == 0
    if not is_valid and raise_on_error:
        raise ManifestVerificationError(f"Cryptographic manifest verification failed:\n" + "\n".join(errors))

    return is_valid, errors


def get_git_commit() -> str | None:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_git_diff_status() -> str | None:
    """Check if the working directory has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return "dirty" if result.stdout.strip() else "clean"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_software_versions() -> dict[str, str]:
    """Capture versions of all relevant software packages."""
    versions: dict[str, str] = {
        "python": sys.version,
        "platform": platform.platform(),
    }

    packages = [
        "stim",
        "pymatching",
        "qiskit",
        "qiskit_ibm_runtime",
        "numpy",
        "scipy",
        "pydantic",
        "matplotlib",
        "torch",
        "adaptive_qec",
    ]

    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    return versions


def capture_reproducibility_info() -> dict[str, Any]:
    """
    Capture all reproducibility information.

    Returns a dict suitable for JSON serialization.
    """
    info: dict[str, Any] = {
        "git_commit": get_git_commit(),
        "git_status": get_git_diff_status(),
        "software_versions": get_software_versions(),
        "python_executable": sys.executable,
    }

    if info["git_status"] == "dirty":
        logger.warning(
            "Working directory has uncommitted changes. "
            "Experiment may not be fully reproducible."
        )

    return info
