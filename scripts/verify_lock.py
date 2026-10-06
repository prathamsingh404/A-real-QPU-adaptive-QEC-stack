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

        actual_hash = compute_sha256(p)
        if actual_hash != expected_hash:
            mismatches.append((rel_path, expected_hash, actual_hash))
            all_matched = False

    if missing:
        logger.error("Missing files tracked in lock: %s", missing)

    if mismatches:
        logger.error("Hash mismatches detected in %d files:", len(mismatches))
        for path, exp, act in mismatches:
            logger.error("  %s: expected %s, got %s", path, exp[:12], act[:12])

    if all_matched:
        logger.info(
            "ALL %d tracked files VERIFIED against SHA-256 lock (commit: %s)",
            len(files),
            lock_data.get("metadata", {}).get("git_commit", "N/A")[:8],
        )
        return True
    else:
        logger.error("Lock verification FAILED.")
        return False


if __name__ == "__main__":
    success = verify_lock()
    sys.exit(0 if success else 1)
