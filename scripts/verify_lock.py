"""Verify the frozen experiment and every scientific artifact listed by the release."""
import json
from pathlib import Path

if __package__:
    from .make_lock import ROOT, check_manifest, compute_sha256, paper_files
else:
    from make_lock import ROOT, check_manifest, compute_sha256, paper_files


def verify_lock(lock_path=Path("paper.lock")):
    check_manifest()
    lock = json.loads((ROOT / lock_path).read_text())
    files = lock["files"]
    if sorted(files) != paper_files():
        print("Lock file set differs from the scientific file list.")
        return False
    for name, expected in files.items():
        if not (ROOT / name).is_file() or compute_sha256(ROOT / name) != expected:
            print(f"Missing or changed artifact: {name}")
            return False
    print(f"Verified {len(files)} scientific artifacts.")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if verify_lock() else 1)
