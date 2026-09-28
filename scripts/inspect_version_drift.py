"""
Analyze environment package versions, uv.lock, and version drift in QEC simulation.
Investigates why pip install gave 1,729 static-MWPM errors vs 1,661 in the artifact.
"""

import json
import re
import sys
from pathlib import Path
import numpy as np
import stim
import pymatching

def inspect_lock():
    lock_path = Path("uv.lock")
    if not lock_path.exists():
        print("uv.lock not found")
        return {}
    
    with open(lock_path, "r", encoding="utf-8") as f:
        text = f.read()

    locked = {}
    pattern = re.compile(r'\[\[package\]\]\s+name = "([^"]+)"\s+version = "([^"]+)"')
    for m in pattern.finditer(text):
        name = m.group(1)
        ver = m.group(2)
        locked[name] = ver
    return locked

def main():
    locked = inspect_lock()
    key_pkgs = ["stim", "pymatching", "numpy", "scipy", "qiskit", "qiskit-ibm-runtime"]
    
    print("=" * 60)
    print("ENVIRONMENT & UV.LOCK COMPARISON")
    print("=" * 60)
    for pkg in key_pkgs:
        installed_ver = "NOT_INSTALLED"
        try:
            mod = __import__(pkg.replace("-", "_"))
            installed_ver = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
        locked_ver = locked.get(pkg, "NOT_IN_LOCK")
        match = "MATCH" if installed_ver == locked_ver else "DRIFT"
        print(f"{pkg:<20} | Installed: {installed_ver:<12} | Locked: {locked_ver:<12} | {match}")

    print("\n" + "=" * 60)
    print("REPRODUCING 1,661 vs 1,729 STATIC MWPM ERRORS")
    print("=" * 60)
    from adaptive_qec.experiments.adaptive_vs_static import run_adaptive_vs_static
    res = run_adaptive_vs_static(distance=3, rounds=3, seed=42)
    mwpm_err = res["arms"]["static_mwpm"]["total_errors"]
    adapt_err = res["arms"]["adaptive"]["total_errors"]
    print(f"Current Installed Environment Output with seed 42:")
    print(f"  Static MWPM Errors: {mwpm_err}")
    print(f"  Adaptive Errors:    {adapt_err}")

if __name__ == "__main__":
    main()
