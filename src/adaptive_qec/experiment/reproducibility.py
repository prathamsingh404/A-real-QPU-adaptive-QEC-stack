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

import logging
import platform
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)


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
