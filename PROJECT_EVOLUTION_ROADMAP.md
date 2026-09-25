# Project Evolution Roadmap: Research-Grade Adaptive QEC on Heavy-Hex Architectures

## From Simulation Prototype to Publishable Hardware-Validated Adaptive Quantum Error Correction

> **Document Status**: ACTIVE SPECIFICATION & RESEARCH BLUEPRINT  
> **Target Target Venues**: *PRX Quantum*, *Nature Communications*, or *IEEE Transactions on Quantum Engineering*  
> **Codebase Target**: `src/adaptive_qec/`  
> **Hardware Target**: IBM Heron Processors (156-qubit Heavy-Hex Lattice, e.g., `ibm_marrakesh`, `ibm_kingston`) via Qiskit Runtime  

---

## Table of Contents

1. [Executive Summary & Central Scientific Thesis](#1-executive-summary--central-scientific-thesis)
2. [Comprehensive Prior-Art Landscape & Novelty Demarcation](#2-comprehensive-prior-art-landscape--novelty-demarcation)
   - 2.1 Deep Taxonomy of Existing Paradigms (2024–2026)
   - 2.2 Detailed Comparative Prior-Art Matrix
   - 2.3 Why We Do Not Copy: Critical Gaps in Prior Art
3. [Theoretical Foundations & Formal Mathematical Claims](#3-theoretical-foundations--formal-mathematical-claims)
   - 3.1 Claim 1: Non-Stationary Multi-Armed Bandit Control Regret Bounds
   - 3.2 Claim 2: Statistically Gated Strategy Switching (Wald SPRT & Wilson Bounds)
   - 3.3 Claim 3: Fault-Tolerance of Dynamic Anisotropic Stabilizer Scheduling (DA-SE)
   - 3.4 Claim 4: Sub-Microsecond Incremental DEM Graph Reweighting
4. [Current Codebase State & Algorithmic Gap Analysis](#4-current-codebase-state--algorithmic-gap-analysis)
5. [End-to-End System Architecture](#5-end-to-end-system-architecture)
6. [Detailed 7-Phase Execution Plan (Weeks 1–17)](#6-detailed-7-phase-execution-plan-weeks-117)
   - [Phase 1: Foundation Hardening, Experiment Harness & Scenarios](#phase-1-foundation-hardening-experiment-harness--scenarios-weeks-12)
   - [Phase 2: Online Bandit Controller & Statistical Decision Framework](#phase-2-online-bandit-controller--statistical-decision-framework-weeks-35)
   - [Phase 3: Dynamic Anisotropic Stabilizer Scheduling](#phase-3-dynamic-anisotropic-stabilizer-scheduling-weeks-57)
   - [Phase 4: Closed-Loop Live DEM & Peeling Graph Calibration](#phase-4-closed-loop-live-dem--peeling-graph-calibration-weeks-79)
   - [Phase 5: Qiskit Runtime Closed-Loop Cloud Architecture](#phase-5-qiskit-runtime-closed-loop-cloud-architecture-weeks-911)
   - [Phase 6: Real QPU Hardware Execution on IBM Heron](#phase-6-real-qpu-hardware-execution-on-ibm-heron-weeks-1114)
   - [Phase 7: Publication, Artifact Release & Open Science Suite](#phase-7-publication-artifact-release--open-science-suite-weeks-1417)
