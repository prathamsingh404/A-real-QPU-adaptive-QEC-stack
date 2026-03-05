"""
Adaptive vs Static Threshold Sweep
Sweeps the physical error rate (p) at distance d=5 to demonstrate that
the adaptive controller achieves a lower logical error rate across the entire
operational range when persistent leakage is present, effectively improving the threshold.
"""

from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
from adaptive_qec.experiments.adaptive_vs_static import run_adaptive_vs_static, NoiseSchedule

def main():
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')
    
    out_dir = Path("experiments/results/threshold")
    out_dir.mkdir(exist_ok=True, parents=True)
    
    d = 5
    # For d=5, num_dets_per_round = 24. We inject > 15% leakage -> ~4 detectors.
    num_dets_per_round = d**2 - 1
    num_leakage = max(2, int(num_dets_per_round * 0.16))
    leak_detectors = list(range(1, num_leakage + 1))
    
    p_values = np.linspace(0.003, 0.015, 7)
