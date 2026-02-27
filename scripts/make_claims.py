#!/usr/bin/env python3
"""
Claims Ledger Generator & Audit Validator (scripts/make_claims.py).

For every claim, number, and metric referenced in repository documentation,
this ledger records:
  - Claim ID & Description
  - Target Artifact Path
  - Command to Reproduce / Validate
  - Git Commit / Tag Hash
  - Documented Claim Value vs Ground-Truth Artifact Value
  - Status: REPRODUCED | DISPROVED | NOT_REPRODUCED | DOWNGRADED | REMOVED
  - Forensic Notes

Enforces zero-drift between committed JSON artifacts and documentation.
Run with `--check` to verify that all active documentation claims match
their underlying artifact JSONs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ClaimEntry:
    claim_id: str
    category: str
    statement: str
    artifact_path: Optional[str]
    artifact_key_path: Optional[str]
    command: str
    expected_doc_value: str
    status: str
    tag_or_commit: str
    notes: str

    def get_artifact_value(self, repo_root: Path) -> tuple[Optional[Any], str]:
        """Read the exact value from the artifact JSON if path exists."""
        if not self.artifact_path:
            return None, "NO_ARTIFACT"
        full_path = repo_root / self.artifact_path
        if not full_path.exists():
            return None, "FILE_NOT_FOUND"
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not self.artifact_key_path:
                return data, "OK"
            curr: Any = data
            for key in self.artifact_key_path.split("."):
                if isinstance(curr, dict) and key in curr:
                    curr = curr[key]
                elif isinstance(curr, list) and key.isdigit():
                    curr = curr[int(key)]
                else:
                    return None, f"KEY_NOT_FOUND: {key}"
            return curr, "OK"
        except Exception as e:
            return None, f"READ_ERROR: {e}"


def get_current_git_hash(repo_root: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "pre-audit-2026-09-30"

