# Telemetry, Observation, and Action Taxonomy for Adaptive QEC

> **Status**: Manuscript Specification & Architectural Boundary Definition  
> **Target Audience**: Reviewers, Experimentalists, and Software Implementers  
> **Relevant Submission Gates**: Gate A (Scientific Correctness)  
> **Tasks Addressed**: TASK-016 (Point 9), TASK-017 (Point 37)

---

## 1. Executive Summary & Boundary Invariant

A fatal flaw in early adaptive QEC literature is **information leakage**: controllers conditioning decisions on unobservable ground truth (such as true latent Pauli error rates $p_X, p_Z$, exact cosmic ray injection schedules, or true logical error rates $P_L$).

In a physical fault-tolerant quantum computer executing an unknown user algorithm, the true logical error rate and instantaneous microscopic noise states are **fundamentally unobservable**. Only classical detector syndromes, macroscopic calibration snapshots, and classical decoder execution statistics can be observed online.

Furthermore, **classical decoder adaptation** (which operates post-measurement on classical bitstreams) must be strictly distinguished from **physical quantum circuit interventions** (which alter pulses, timing, or basis scheduling on the physical qubits).

---

## 2. Telemetry Observability Taxonomy Table
