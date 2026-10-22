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
    
    for p in p_values:
        print(f"\n{'='*80}")
        print(f"RUNNING SWEEP FOR p_phys={p:.4f} at d={d}")
        print(f"{'='*80}")
        
        # We use a constant p for this sweep point, but with leakage starting at window 5.
        sched = NoiseSchedule(
            total_windows=30,
            shots_per_window=500, # 15000 shots per arm per point
            drift_start_window=100, # no drift
            drift_end_window=101,
            drift_p2q_base=float(p),
            drift_p2q_peak=float(p),
            burst_windows=[], # no bursts, just pure background noise + leakage
            burst_intensity=0.0,
            leakage_start_window=5,
            leakage_detector_indices=leak_detectors,
        )
        
        run_adaptive_vs_static(
            distance=d,
            rounds=d,
            schedule=sched,
            output_dir=out_dir,
            seed=42 + int(p*10000),
        )

if __name__ == "__main__":
    main()
