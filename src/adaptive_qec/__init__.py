"""
AdaptiveQEC - An experimental platform for hardware-aware, adaptive,
low-latency quantum error correction on real quantum processing units.

Integrates real-time calibration ingest, non-stationary noise drift scenarios,
online multi-armed bandit controllers (Exp3, Exp3.P, DA-SE, SPRT),
dynamic DEM edge reweighting, adaptive X/Z stabilizer scheduling,
Qiskit Runtime closed-loop execution, shot budget enforcement,
and rigorous statistical regret evaluation.

Core packages:
    controller:   Bandit-based adaptive strategy selection
    qec:          Circuit generation and stabilizer scheduling
    decoders:     MWPM and Union-Find with dynamic weight updates
    runtime:      Qiskit Runtime session management and budget
    experiments:  Reproducible experiment scripts
    analysis:     Statistical significance and regret analysis
    noise:        Drift detection, burst detection, leakage tracking
    mitigation:   Dynamical decoupling integration
"""

__version__ = "0.3.0"
__project__ = "AdaptiveQEC"

