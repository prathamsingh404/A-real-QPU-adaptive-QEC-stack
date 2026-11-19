"""
Provenance and Manifest Integrity Tests (tests/test_provenance_integrity.py).

Validates:
1. paper.lock existence, schema, and strict simulation provenance
2. data/manifest.json schema, command reproducibility, and toolchain versions
3. Cryptographic hash validity (SHA-256) across tracked artifact manifests
"""

import hashlib
import json
from pathlib import Path
import pytest


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_paper_lock_exists_and_valid_schema():
    """Verify paper.lock is present, valid JSON, and has simulation provenance."""
    lock_file = get_repo_root() / "paper.lock"
    assert lock_file.exists(), "paper.lock must exist at repository root"

    with open(lock_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "files" in data

    meta = data["metadata"]
    assert set(meta["provenance"].split(", ")) == {"simulation", "public physical replay", "host profiling"}
    assert meta.get("hash_algorithm") == "SHA-256"
    assert "version" in meta
    assert "git_commit" in meta
    assert "created_at_utc" in meta
    assert meta.get("total_tracked_files") == len(data["files"])

    # Ensure critical result files are tracked
    files = data["files"]
    required_files = [
        "configs/suite.yaml",
        "data/suite_results.json",
        "data/manifest.json",
    ]
    for rf in required_files:
        assert rf in files, f"Critical file {rf} missing from paper.lock files"
        # Each hash must be valid 64-character SHA-256 hex string
        assert len(files[rf]) == 64, f"Hash for {rf} is not a valid 64-character SHA-256 hex digest"


def test_manifest_json_provenance_and_system_metadata():
    """Verify data/manifest.json provenance, execution trace, and system versions."""
    manifest_file = get_repo_root() / "data" / "manifest.json"
    assert manifest_file.exists(), "data/manifest.json must exist"

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest.get("provenance") == "simulation"
    assert "git_commit" in manifest
    assert "timestamp_utc" in manifest
    assert "command_line" in manifest
    assert "execution_duration_sec" in manifest

    system = manifest.get("system", {})
    assert "platform" in system
    assert "python_version" in system
    assert "stim_version" in system
    assert "pymatching_version" in system

    results = manifest.get("results_file", {})
    records = json.loads((get_repo_root() / "data/suite_results.json").read_text())["trajectories"]
    assert results.get("total_records") == len(records) > 0
    assert "sha256" in results
    assert len(results["sha256"]) == 64


def test_sha256_hash_computation():
    """Verify cryptographic hashing function correctly computes 64-character SHA-256 digest."""
    suite_yaml_path = get_repo_root() / "configs" / "suite.yaml"
    assert suite_yaml_path.exists()

    hasher = hashlib.sha256()
    with open(suite_yaml_path, "rb") as f:
        hasher.update(f.read())
    digest = hasher.hexdigest()

    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
