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
