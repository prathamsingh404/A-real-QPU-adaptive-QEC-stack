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
