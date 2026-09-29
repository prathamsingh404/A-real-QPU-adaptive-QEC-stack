"""
Generate Cryptographic SHA-256 Provenance Lock (paper.lock).

Computes real SHA-256 hashes for all empirical simulation data files,
generated manuscript tables, and figures, binding them to the git commit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MakeLock")

TRACKED_FILES = [
    Path("configs/suite.yaml"),
    Path("data/suite_results.json"),
    Path("data/manifest.json"),
    Path("papers/tables/table_1.tex"),
    Path("papers/tables/table_2.tex"),
    Path("papers/tables/table_3.tex"),
    Path("papers/tables/table_4.tex"),
    Path("papers/figures/fig_1_window_variance_tradeoff.pdf"),
    Path("papers/figures/fig_2_architecture_diagram.pdf"),
    Path("papers/figures/fig_4_distance_scaling.pdf"),
    Path("papers/figures/fig_5_paired_deltas.pdf"),
]


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def generate_lock(lock_path: Path = Path("paper.lock")) -> None:
    """Generate paper.lock with genuine SHA-256 digests."""
    git_commit = get_git_commit()
    file_hashes: Dict[str, str] = {}

    missing_files = []
    for f in TRACKED_FILES:
        if not f.exists():
            missing_files.append(str(f))
        else:
            file_hashes[str(f).replace("\\", "/")] = compute_sha256(f)

    if missing_files:
        logger.warning("Some tracked files do not exist yet: %s", missing_files)

    lock_payload = {
        "metadata": {
            "version": "2.0.0",
            "git_commit": git_commit,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "hash_algorithm": "SHA-256",
            "provenance": "simulation",
            "total_tracked_files": len(file_hashes),
        },
        "files": file_hashes,
    }

    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(lock_payload, f, indent=2)

    logger.info("Successfully generated %s with %d file hashes (commit: %s)", lock_path, len(file_hashes), git_commit[:8])


if __name__ == "__main__":
    generate_lock()
