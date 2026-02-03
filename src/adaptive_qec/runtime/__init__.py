"""
Runtime package: QPU execution, budget management, and profiling.

Modules:
    qiskit_loop: Closed-loop batched Qiskit Runtime session driver
    budget: Shot budget manager with quota enforcement
    profiler: Latency and throughput profiling
"""

from adaptive_qec.runtime.budget import BudgetConfig, ShotBudgetManager
