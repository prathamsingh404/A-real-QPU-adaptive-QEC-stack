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
7. [Hardware Experimentation Protocol on IBM Heron](#7-hardware-experimentation-protocol-on-ibm-heron)
8. [Statistical Rigor, Power Analysis & Ablation Protocol](#8-statistical-rigor-power-analysis--ablation-protocol)
9. [File-Level Implementation Map](#9-file-level-implementation-map)
10. [Risk Register & Physical Failure Mode Mitigations](#10-risk-register--physical-failure-mode-mitigations)
11. [Traceability & Dependency Matrix](#11-traceability--dependency-matrix)

---

## 1. Executive Summary & Central Scientific Thesis

### 1.1 The Fundamental Problem
Standard Quantum Error Correction (QEC) protocols universally operate under the assumption of **stationary, Markovian, and isotropic noise**. Circuits, stabilizer measurement sequences, and decoding graphs (such as Minimum-Weight Perfect Matching or Union-Find decoding hypergraphs) are compiled *offline* based on nominal factory calibration sheets or ideal phenomenological models.

However, physical superconducting quantum processing units (QPUs) violate all three assumptions:
1. **Non-Stationarity**: Coherence times ($T_1, T_2$), two-qubit gate fidelities, and readout misidentifications fluctuate wildly over hours due to two-level fluctuator (TLF) spectral diffusion and thermal drifts.
2. **Anisotropy / Noise Asymmetry**: Energy relaxation ($T_1$) creates pronounced asymmetry between bit-flip ($X$) and phase-flip ($Z$) errors, which shifts dynamically as individual qubits degrade or recover.
3. **Correlated Spatial-Temporal Bursts**: Stray high-energy ionizing radiation (cosmic rays, substrate gamma rays) and quasiparticle poisoning induce localized spatio-temporal error bursts that violate independent edge weight assumptions in decoders.

### 1.2 Central Scientific Thesis
> **Hypothesis**: An online, data-driven closed-loop controller that couples:
> 1. **Empirical Multi-Armed Bandit (MAB) learning** over discrete decoding and dynamical decoupling actions,
> 2. **Sequential hypothesis testing (Wald SPRT / Wilson bounds)** to prevent spurious strategy oscillation,
> 3. **Dynamic Anisotropic Stabilizer Scheduling (DA-SE)** responding to real-time syndrome bias, and
> 4. **Live closed-loop Detector Error Model (DEM) graph reweighting**,
> 
> will achieve a **statistically significant reduction in logical error rate (LER)** over static, factory-calibrated QEC baselines under physical noise drift on IBM heavy-hex superconducting hardware, while incurring **sub-microsecond classical decision overhead**, without requiring neural network accelerators or millions of offline training samples.

---

## 2. Comprehensive Prior-Art Landscape & Novelty Demarcation

To ensure our work represents a groundbreaking, peer-reviewable contribution and does not inadvertently duplicate existing open-source projects, industrial demos, or academic papers, we conducted a rigorous literature and patent/repository survey across quantum computing and machine learning venues from 2021 through 2026.

### 2.1 Deep Taxonomy of Existing Paradigms (2024–2026)

#### A. Deep Neural Network Decoders & Transformers
*   **AlphaQubit & AlphaQubit 2** (Google DeepMind & Google Quantum AI, *Nature* 635, 2024; arXiv:2512.07737, Dec 2025):
    *   *Approach*: Large transformer/recurrent architectures trained via supervised and reinforcement learning on millions of simulated and hardware-generated syndrome shots on the Sycamore processor.
    *   *Strengths*: Exceptional decoding accuracy near the theoretical maximum likelihood limit; handles complex correlated noise.
    *   *Critical Weaknesses & Gaps*: Massive computational footprint (requires TPU/GPU inference servers); inference latency ($>100\,\mu\text{s}$ to milliseconds per round in early iterations, scaled down via custom quantization); complete black box; static once weights are frozen (incapable of real-time online adaptation to out-of-distribution drift without costly offline fine-tuning).
    *   *Our Delineation*: We do not build another neural decoder. We use lightweight classical decoders (PyMatching 2, radius-weighted Union-Find) and adapt their mathematical graph parameters online with sub-microsecond latency.

#### B. Continual Learning & Neural Pre-Decoding
*   **QAdapt** (arXiv:2607.28422, July 2026) & **SAGE-QEC** (2026):
    *   *Approach*: Neural pre-decoders that ingest syndrome histories to capture local spatio-temporal correlations, forwarding residual syndromes to a global matching decoder. Uses continual learning to handle noise distribution shifts.
    *   *Critical Weaknesses & Gaps*: Susceptible to catastrophic forgetting; lacks formal convergence guarantees; adds classical latency; does not alter physical quantum control (no dynamical decoupling selection, no adaptive stabilizer scheduling).
    *   *Our Delineation*: Our adaptation occurs at both the *circuit control layer* (dynamical decoupling sequences, stabilizer check ratios) and the *algorithmic decoder layer* (DEM edge reweighting), backed by provable sub-linear bandit regret bounds rather than heuristic neural loss optimization.

#### C. Reinforcement Learning & Bandit Retraining for Variational Codes
*   **BRAVE: Bandit Retraining for Adaptive Variational Error Correction** (Guatto, Preti, Schilling, Calarco, Cárdenas-López, Motzoi, arXiv:2509.03974, July 2026):
    *   *Approach*: Combines offline Multi-Agent Reinforcement Learning (MARL) to discover variational quantum error correcting circuits with an online bandit layer that decides when to trigger retraining of low-dimensional variational angles.
    *   *Critical Weaknesses & Gaps*: Restricted to small, unencoded, continuous variational parameterizations (qubits/qutrits); does not apply to topological stabilizer codes (surface codes, heavy-hex codes) where stabilizer checks are discrete projections; relies on continuous gate re-parameterization which is incompatible with fixed native cross-resonance gate calibrations on IBM cloud hardware.
    *   *Our Delineation*: We formulate an online bandit controller explicitly over *discrete combinatorial action spaces* in topological stabilizer codes on standard superconducting transmon architectures.

#### D. Passive Noise Estimation & Sliding-Window Filtering
*   **Bhardwaj, Takou, Lin, and Brown** (Duke University, *PRX Quantum* 7, 033024, August 2026; arXiv:2511.09491):
    *   *Approach*: "Adaptive Estimation of Drifting Noise in Quantum Error Correction." Uses an analytical sliding-window and overlapping spectral filter to passively recover time-dependent Pauli noise frequency components directly from syndrome statistics.
    *   *Critical Weaknesses & Gaps*: Strictly **passive** characterization. They estimate the drifting noise parameters in simulation, but do not close the control loop: no dynamic strategy switching, no adaptive stabilizer scheduling, no runtime hardware control.
    *   *Our Delineation*: We take syndrome-derived noise tracking and close the active feedback loop on real QPUs: updating PyMatching DEM graphs online, adjusting X/Z measurement ratios, and switching dynamical decoupling sequences.

#### E. Concatenated Adaptive Syndrome Extraction
*   **Berthusen, Tan, Huang, and Gottesman** (*PRX Quantum* 6, 030307, 2025; arXiv:2502.14835):
    *   *Approach*: "Adaptive Syndrome Extraction." Demonstrates that in concatenated codes (specifically a $[[4,2,2]]$ code concatenated with hypergraph product codes), measuring physical error detectors in the inner code can selectively prune which outer stabilizer generators need to be measured.
    *   *Critical Weaknesses & Gaps*: Requires a specific concatenated code structure; inapplicable to 2D topological planar or heavy-hex surface codes where stabilizer generators must be measured periodically to form continuous spacetime fault paths.
    *   *Our Delineation*: We implement **Dynamic Anisotropic Stabilizer Scheduling (DA-SE)** directly on heavy-hex surface codes, varying the global temporal measurement ratio between non-commuting check operators ($X$ vs $Z$) in response to measured physical noise bias, without changing the code code-family.

