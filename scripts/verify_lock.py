"""
Verify Cryptographic SHA-256 Provenance Lock (paper.lock).

Recomputes SHA-256 digests for all tracked files and verifies that they
match the hashes recorded in paper.lock. Fails with non-zero exit code on mismatch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VerifyLock")


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_lock(lock_path: Path = Path("paper.lock")) -> bool:
    """Verify all hashes in paper.lock."""
    if not lock_path.exists():
        logger.error("Lock file not found: %s", lock_path)
        return False

    with open(lock_path, "r", encoding="utf-8") as f:
        lock_data = json.load(f)

    files: dict[str, str] = lock_data.get("files", {})
    if not files:
        logger.error("Lock file contains no tracked file hashes.")
        return False

    all_matched = True
    mismatches = []
    missing = []

    for rel_path, expected_hash in files.items():
        p = Path(rel_path)
        if not p.exists():
            missing.append(rel_path)
            all_matched = False
            continue
