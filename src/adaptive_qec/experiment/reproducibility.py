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
