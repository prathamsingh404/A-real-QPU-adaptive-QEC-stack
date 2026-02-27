"""
Multi-Distance Adaptive vs Static Sweep
Runs the experiment at d=3, d=5, and d=7 to demonstrate the distance-scaling
and threshold properties of the adaptive controller under persistent leakage.
"""

from __future__ import annotations
import logging
from pathlib import Path
from adaptive_qec.experiments.adaptive_vs_static import run_adaptive_vs_static, NoiseSchedule

def main():
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')
    
    out_dir = Path("experiments/results")
    out_dir.mkdir(exist_ok=True, parents=True)
    
    for d in [3, 5, 7]:
        print(f"\n{'='*80}")
        print(f"RUNNING SWEEP FOR DISTANCE d={d}")
        print(f"{'='*80}")
        
        # We need the leakage fraction to be > 0.15 for the fast-path to trigger.
        num_dets_per_round = d**2 - 1
        num_leakage = max(2, int(num_dets_per_round * 0.16))
        leak_detectors = list(range(1, num_leakage + 1))
