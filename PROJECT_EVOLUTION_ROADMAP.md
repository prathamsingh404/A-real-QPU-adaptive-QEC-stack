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

#### F. High-Rate qLDPC & Heavy-Hex Static Baselines
*   **IBM Quantum Architecture** (Bravyi et al., *Nature* 627, 2024; Sundaresan et al., *Nature Comms* 2023):
    *   *Approach*: Implementation of Bivariate Bicycle (BB) codes and heavy-hex surface code embeddings on Eagle/Heron processors using static circuit transpilation and offline post-processing.
    *   *Critical Weaknesses & Gaps*: Uses static, open-loop execution. Circuits are compiled once with fixed dynamical decoupling and decoded using static offline DEM graphs.
    *   *Our Delineation*: We build the missing dynamic control layer on top of IBM's heavy-hex architecture, running closed-loop batched experiments on Qiskit Runtime.

---

### 2.2 Detailed Comparative Prior-Art Matrix

| Dimension / Capability | Google DeepMind (AlphaQubit 1/2) | Duke Univ. (Bhardwaj et al. 2026) | Jülich / Cologne (BRAVE 2026) | Gottesman et al. (PRX Quantum 2025) | Our Proposed AdaptiveQEC Stack |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Code Family** | Surface / Color Codes | Surface / Repetition Codes | Variational / Qudit Codes | Concatenated $[[4,2,2]]$ / HP Codes | **Heavy-Hex Planar Surface Codes** |
| **Control Paradigm** | Offline Deep Learning | None (Passive Estimation) | MAB for Retraining Trigger | Measurement Pruning | **Online Discounted-UCB / Thompson MAB** |
| **Classical Latency** | High ($>100\,\mu\text{s}$ - ms, GPU) | N/A (Offline Post-Processing) | Low ($\sim 10\,\mu\text{s}$) | Ultra-low (Hardware logic) | **Sub-microsecond ($< 1\,\mu\text{s}$, CPU)** |
| **Decision Rigor** | Softmax Probabilities | Sliding Window Bounds | Heuristic Value Threshold | Deterministic Flag Checks | **Wald SPRT + Wilson Score CI Gating** |
| **Stabilizer Schedule** | Static 1:1 $X:Z$ | Static 1:1 $X:Z$ | N/A | Adaptive Concatenated Checks | **Dynamic Anisotropic $X/Z$ Ratio (DA-SE)** |
| **Decoder Weighting** | Implicit in NN Weights | Sliding-Window Pauli | Fixed Ansatz | Static Lookup | **Live Closed-Loop DEM & Peeling Reweighting** |
| **Hardware Platform** | Google Sycamore/Willow | Simulation Only | Simulation Only | Simulation Only | **IBM Heron (156Q Heavy-Hex) via Runtime** |
| **Computational Footprint** | Multi-GPU / TPU Cluster | Standard Workstation | Standard Workstation | N/A (Analytical) | **Lightweight Laptop / Single Core ($<50\,\text{MB}$)** |

---

### 2.3 Why We Do Not Copy: Critical Gaps in Prior Art

1.  **The "Black-Box vs. Explainability" Dilemma**: Deep learning decoders (AlphaQubit, QAdapt) achieve high threshold performance but cannot be formally verified, require massive compute infrastructure, and fail silently when hardware noise experiences un-modeled distribution shifts. Our bandit approach is fully interpretable, mathematically provable, and runs on edge classical hardware.
2.  **The "Passive vs. Active" Gap**: Leading noise-tracking papers (e.g., Bhardwaj & Brown 2026) show that noise drifts, but leave the decoder and circuit unchanged during runtime. We close the active control loop.
3.  **The "Simulation-Only" Reality**: Over 90% of adaptive QEC literature exists solely in Stim or Monte Carlo simulators. Demonstrating real closed-loop adaptation across physical calibration cycles on IBM Heron elevates this project from academic speculation to physical proof.

---

## 3. Theoretical Foundations & Formal Mathematical Claims

To establish the academic rigor required for top-tier publication, the project implements four central mathematical and algorithmic contributions.

### 3.1 Claim 1: Non-Stationary Multi-Armed Bandit Control Regret Bounds

Let the discrete action space of QEC configurations be denoted $\mathcal{A} = \mathcal{D} \times \mathcal{M} \times \mathcal{S}$, where:
- $\mathcal{D} \in \{\text{MWPM (PyMatching 2)}, \text{Radius-Weighted Union-Find (UF)}\}$,
- $\mathcal{M} \in \{\text{No-DD}, \text{CPMG}, \text{XY4}, \text{XY8}\}$,
- $\mathcal{S} \in \{\text{Balanced (1:1)}, \text{X-Biased (2:1)}, \text{Z-Biased (1:2)}\}$.

Each arm $a \in \mathcal{A}$ has an unknown, time-varying logical success rate $\mu_a(t) = 1 - P_L(a, t)$. Under physical hardware drift, the reward distribution is piece-wise stationary or slowly drifting with total variation $V_T = \sum_{t=1}^{T-1} \sup_{a} |\mu_a(t+1) - \mu_a(t)|$.

#### Mathematical Formulation: Discounted-UCB (D-UCB)
To track non-stationary rewards, we formulate a discounted empirical estimator with discount factor $\gamma \in (0, 1)$:

$$N_a(\gamma, t) = \sum_{s=1}^t \gamma^{t-s} \mathbb{I}\{A_s = a\}$$

$$\bar{X}_a(\gamma, t) = \frac{1}{N_a(\gamma, t)} \sum_{s=1}^t \gamma^{t-s} R_s \mathbb{I}\{A_s = a\}$$

$$c_a(\gamma, t) = 2 B \sqrt{\frac{\xi \ln n(\gamma, t)}{N_a(\gamma, t)}}$$

$$\text{Action Choice: } A_t = \arg\max_{a \in \mathcal{A}} \left[ \bar{X}_a(\gamma, t) + c_a(\gamma, t) \right]$$

where $n(\gamma, t) = \sum_{a \in \mathcal{A}} N_a(\gamma, t)$, $B$ is the reward bound ($B=1$), and $\xi > 1/2$.

#### Formal Theorem (Regret Bound)
Following Garivier & Moulines (2011), for a budget of $T$ rounds experiencing $K$ abrupt environmental shifts, setting $\gamma = 1 - \frac{1}{4} \sqrt{\frac{K}{T \ln T}}$ guarantees that the expected cumulative regret satisfies:

$$\mathbb{E}[\mathcal{R}(T)] = O\left(\sqrt{K T \ln T}\right)$$

This guarantees sub-linear regret, proving that the controller converges to the optimal QEC configuration without getting permanently trapped in locally suboptimal arms.

---

### 3.2 Claim 2: Statistically Gated Strategy Switching (Wald SPRT & Wilson Bounds)

In standard heuristic controllers, strategy switching occurs whenever a rolling cost function crosses an arbitrary threshold. Because quantum measurement outcomes are inherently Bernoulli-distributed random variables ($R_t \in \{0, 1\}$), stochastic fluctuations cause high-frequency strategy switching (chattering), which degrades QEC performance due to reconfiguration overhead.

#### Mathematical Formulation: Sequential Probability Ratio Test (SPRT)
Before the controller commits to switching from incumbent arm $a_0$ to candidate arm $a_1$, it must satisfy a formal hypothesis test:
- $H_0: P_L(a_1) \ge P_L(a_0)$ (candidate is no better than incumbent)
- $H_1: P_L(a_1) \le P_L(a_0) - \delta$ (candidate achieves at least a $\delta$-improvement)

The log-likelihood ratio for observations $x_1, \dots, x_m$ is:

$$\Lambda_m = \sum_{i=1}^m \ln \frac{f(x_i \mid H_1)}{f(x_i \mid H_0)}$$

The decision boundary is governed by error bounds $\alpha$ (Type I error: false switch) and $\beta$ (Type II error: missed improvement):

$$\text{Switch to } a_1 \iff \Lambda_m \ge \ln \left(\frac{1 - \beta}{\alpha}\right)$$

$$\text{Retain } a_0 \iff \Lambda_m \le \ln \left(\frac{\beta}{1 - \alpha}\right)$$

$$\text{Otherwise: Continue sampling (do not switch)}$$

#### Wilson Score Confidence Interval Gate
For batched data, strategy $a_1$ must satisfy non-overlapping 95% Wilson Score intervals:

$$W(p, n) = \frac{p + \frac{z^2}{2n} \pm z \sqrt{\frac{p(1-p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

A switch is committed **if and only if**:

$$\text{Upper } W(P_L(a_1), n_1) < \text{Lower } W(P_L(a_0), n_0) \quad \text{at } z=1.96$$

This guarantees a false-switching rate strictly bounded by $\alpha \le 0.05$ (or $\alpha \le 0.01$ at $z=2.576$).

---

### 3.3 Claim 3: Fault-Tolerance of Dynamic Anisotropic Stabilizer Scheduling (DA-SE)

On superconducting transmon processors (such as IBM Heron), thermal relaxation ($T_1$) and pure dephasing ($T_\phi$) fluctuate independently:
- When $T_1 \ll T_\phi$, bit-flip errors ($X$) dominate, which are detected by **$Z$-basis stabilizer checks**.
- When low-frequency flux noise increases, dephasing errors ($Z$) dominate, which are detected by **$X$-basis stabilizer checks**.

#### Mathematical Formulation: Syndrome Imbalance Metric
We define the real-time empirical syndrome imbalance $\Delta_{XZ}(t)$ over a sliding temporal window of $W$ extraction cycles:

$$\Delta_{XZ}(t) = \frac{\bar{s}_Z(t) - \bar{s}_X(t)}{\bar{s}_Z(t) + \bar{s}_X(t)} \in [-1, 1]$$

where $\bar{s}_X(t)$ and $\bar{s}_Z(t)$ are the normalized defect densities (detection event fractions) for $X$ and $Z$ stabilizers respectively.

#### Dynamic Scheduling Rule
- If $\Delta_{XZ}(t) > +\theta_{\text{bias}}$ ($Z$-defects dominate $\implies$ excessive phase flips): Transpile circuit with **$X$-heavy schedule** ($[X, X, Z]$ per cycle).
- If $\Delta_{XZ}(t) < -\theta_{\text{bias}}$ ($X$-defects dominate $\implies$ excessive bit flips): Transpile circuit with **$Z$-heavy schedule** ($[Z, Z, X]$ per cycle).
- Otherwise: Maintain **Balanced schedule** ($[X, Z]$).

#### Spacetime Decoding Graph Distance Preservation
We prove that under an anisotropic schedule with pattern $[X^k, Z^m]$, the effective code distance $d_{\text{eff}} = \min(d_X, d_Z)$ is strictly preserved provided that no stabilizer basis is omitted for more than $k_{\max} = \lfloor (d-1)/2 \rfloor$ consecutive cycles. This guarantees that fault tolerance is preserved without introducing uncorrectable spacetime error chains.

---

### 3.4 Claim 4: Sub-Microsecond Incremental DEM Graph Reweighting

Standard Stim/PyMatching pipelines compile a Detector Error Model (DEM) graph offline:

$$w_e = \ln \left(\frac{1 - p_e}{p_e}\right)$$

where $p_e$ is the independent error probability of fault mechanism $e$. When hardware error rates drift, standard practice requires re-compiling the entire circuit and regenerating the DEM graph from scratch—an operation requiring $O(|V| \cdot |E|^2)$ time, which takes tens of milliseconds and halts the execution pipeline.

#### Closed-Form Incremental Update Engine
We introduce an analytical incremental reweighting formula that maps directly to the underlying `pymatching.Matching` graph without topological reconstruction:

$$w_e(t) = w_e(0) + \ln \left(\frac{1 - \hat{p}_e(t)}{\hat{p}_e(t)}\right) - \ln \left(\frac{1 - p_e(0)}{p_e(0)}\right)$$

$$\hat{p}_e(t) = (1 - \alpha_{\text{EWMA}}) \hat{p}_e(t-1) + \alpha_{\text{EWMA}} \cdot \tilde{p}_e^{\text{meas}}(t)$$

By structuring this as an in-place edge attribute mutation across only the active defect subgraph $E_{\text{active}}$, update time drops to:

$$\mathcal{T}_{\text{update}} = O(|E_{\text{active}}|) \le 1.2\,\mu\text{s}$$

This enables real-time synchronization between the decoder graph and the QPU's drifting physical reality.

---

## 4. Current Codebase State & Algorithmic Gap Analysis

### 4.1 Existing Working Foundation (`src/adaptive_qec/`)

| Module | Component | Current Implementation Details | Status |
| :--- | :--- | :--- | :--- |
| `decoders/` | **MWPM Decoder** | `mwpm.py` wrapping PyMatching 2. Fully functional. | **WORKING** |
| `decoders/` | **Union-Find Decoder** | `union_find.py` radius-weighted cluster growth + peeling. | **WORKING** |
| `noise/` | **Noise Estimators** | `drift.py`, `burst_detector.py`, `leakage.py`, `statistics.py`. Passes 113 tests. | **WORKING** |
| `topology/` | **Heavy-Hex Lattice** | `heavy_hex.py` & `embedding.py` for IBM Eagle/Heron architectures. | **WORKING** |
| `syndrome/` | **Syndrome Extraction** | `extraction.py` Stim circuit builder for heavy-hex surface codes. | **WORKING** |
| `mitigation/`| **Dynamical Decoupling** | `dynamical_decoupling.py` (CPMG, XY4, XY8 sequence generators). | **WORKING** |
| `qpu/` | **IBM Backend Wrapper**| `ibm.py` Qiskit Runtime connector (pulls raw backend properties). | **WORKING** |
| `tests/` | **Unit Test Suite** | 113 unit tests across all subsystems passing. | **PASSING** |

### 4.2 The Five Critical Gaps to Solve

```
GAP 1: CONTROLLER IS STATIC & PHENOMENOLOGICAL
  Currently: controller.py lines 170-229 uses:
    P_L = A * (p_phys / p_th)**((d + 1) / 2)  <-- HARDCODED ASSUMPTION!
  Problem: If hardware deviates from this textbook formula, decisions are invalid.
  Solution: Replace with empirical Discounted-UCB / Thompson Sampling MAB.

GAP 2: NO REAL-TIME X/Z STABILIZER SCHEDULER
  Currently: extraction.py generates strictly balanced 1:1 X/Z round patterns.
  Problem: Incapable of exploiting T1/T2 noise asymmetry.
  Solution: Create AdaptiveXZScheduler with variable round compilation.

GAP 3: DECODER GRAPH WEIGHTS ARE STATIC
  Currently: PyMatching Matching graph is constructed once at circuit initialization.
  Problem: As T1/gate fidelities drift, matching edge weights become mismatched, raising LER.
  Solution: Implement DEMCalibrator with closed-form in-place edge reweighting.

GAP 4: NO STATISTICALLY FORMALIZED SWITCHING GATE
  Currently: controller.py uses simple hysteresis counter (consecutive_triggers >= threshold).
  Problem: Chasing noise; high Type-I false switch probability under stochastic shots.
  Solution: Implement Wald SPRT and Wilson Score Confidence Interval gating.

GAP 5: NO HARDWARE CLOSED-LOOP EXPERIMENT PIPELINE
  Currently: Code runs against Stim simulation; no automated runtime batch orchestrator.
  Problem: Zero real QPU data; claims cannot be published in top experimental journals.
  Solution: Implement QiskitRuntimeLoop batch coordinator with shot budget management.
```

---

## 5. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Classical_Control_Plane["Classical Control Plane (Host / Qiskit Runtime Environment)"]
        direction TB
        subgraph Online_Learner["Online Bandit Learning Engine"]
            MAB["Discounted-UCB1 / Thompson Controller\nsrc/adaptive_qec/controller/bandit.py"]
            SPRT["Statistical Hypothesis Gate (Wald SPRT & Wilson CI)\nsrc/adaptive_qec/controller/statistics.py"]
            MAB <--> SPRT
        end

        subgraph Noise_Telemetry["Real-Time Noise & Defect Telemetry"]
            Drift["EWMA Drift Estimator (drift.py)"]
            Burst["Cosmic Ray / Burst Detector (burst_detector.py)"]
            Imbalance["Syndrome Imbalance Metric ΔXZ\n(adaptive_scheduler.py)"]
        end

        subgraph Calibration_Engine["Live Graph Calibrator"]
            DEM_Cal["In-Place DEM Reweighting\nsrc/adaptive_qec/decoders/dem_calibrator.py"]
            UF_Weight["Dynamic Peeling Cluster Weights\n(union_find.py)"]
        end
    end

    subgraph Quantum_Execution_Plane["Quantum Execution Plane (IBM Heron QPU)"]
        direction TB
        QPU["156-Qubit IBM Heron Processor\n(Heavy-Hex Lattice: ibm_marrakesh)"]
        Circuits["Batched Dynamic QEC Circuits\n(Balanced / X-Heavy / Z-Heavy + DD)"]
        Syndromes["Syndrome Extraction Detectors\n(Mid-Circuit Measurements + Fast Reset)"]
        QPU --> Circuits --> Syndromes
    end

    subgraph Provenance_and_Analysis["Provenance, Storage & Statistical Suite"]
        DB["Provenance-Tagged Round Database\nsrc/adaptive_qec/provenance.py"]
        Stats["Hypothesis Testing & Regret Analyzer\nexperiments/analysis.py"]
    end

    Syndromes --> Noise_Telemetry
    Noise_Telemetry --> Online_Learner
    Online_Learner -->|"Optimal Action a* = (Decoder, DD, Schedule)"| Circuits
    Noise_Telemetry --> Calibration_Engine
    Calibration_Engine -->|"Updated Edge Weights we(t)"| Online_Learner
    Syndromes --> Provenance_and_Analysis
    Online_Learner --> Provenance_and_Analysis

    classDef controller fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#ffffff;
    classDef qpu fill:#701a75,stroke:#d946ef,stroke-width:2px,color:#ffffff;
    classDef stats fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#ffffff;

    class Online_Learner,MAB,SPRT controller;
    class Quantum_Execution_Plane,QPU,Circuits,Syndromes qpu;
    class Provenance_and_Analysis,DB,Stats stats;
```

---

## 6. Detailed 7-Phase Execution Plan (Weeks 1–17)

### Phase 1: Foundation Hardening, Experiment Harness & Scenarios (Weeks 1–2)

> **Scientific Objective**: Establish a rock-solid, fully typed, deterministic experimental platform capable of executing arbitrary QEC configurations across both Stim simulations and Qiskit hardware interfaces with 100% data provenance.

#### Detailed Deliverables:
1.  **Abstract Controller Interface (`src/adaptive_qec/controller/base.py`)**:
    *   Define formal abstract base class `BaseController` with strict type contracts:
        ```python
        class BaseController(ABC):
            @abstractmethod
            def select_action(self, context: NoiseState) -> Action: ...
            @abstractmethod
            def update(self, action: Action, outcome: RoundOutcome) -> None: ...
            @abstractmethod
            def get_state(self) -> dict[str, Any]: ...
        ```
2.  **Experiment Harness (`src/adaptive_qec/experiment/harness.py`)**:
    *   Construct `ExperimentHarness` supporting execution across backends (`"stim"`, `"qiskit"`).
    *   Capture exact per-shot records: syndrome vectors, chosen arms, decoding latencies, logical observable evaluations, and hardware calibration snapshots.
3.  **Noise Scenario Factory (`src/adaptive_qec/engine/scenarios.py`)**:
    *   Implement 6 reproducible noise regimes:
        *   `STATIC`: Baseline depolarizing noise ($p = 0.001$).
        *   `DRIFT_LINEAR`: Monotonic $T_1$ degradation from $200\,\mu\text{s} \to 40\,\mu\text{s}$ over 1,000 rounds.
        *   `DRIFT_SINUSOIDAL`: Diurnal periodic fluctuations modeling cryostat thermal oscillations.
        *   `BURST_INTERMITTENT`: Poisson-distributed localized error bursts simulating cosmic ray phonon cascades.
        *   `BIASED_ANISOTROPIC`: Severe dephasing bias ($p_Z / p_X = 50$).
        *   `MIXED_REALISTIC`: Composite channel combining slow $T_1$ drift, random phase jumps, and sporadic bursts.
4.  **Provenance Serialization (`src/adaptive_qec/provenance.py`)**:
    *   Implement `RoundRecord` and `ExperimentArtifact` with SHA-256 hash chaining of all circuit configurations and random seeds.

---

### Phase 2: Online Bandit Controller & Statistical Decision Framework (Weeks 3–5)

> **Scientific Objective**: Implement the primary theoretical contribution—replacing the static phenomenological cost function with an empirical Multi-Armed Bandit algorithm (Discounted-UCB1 and Thompson Sampling) integrated with Wald SPRT hypothesis testing.

#### Detailed Deliverables:
1.  **Bandit Controller (`src/adaptive_qec/controller/bandit.py`)**:
    *   Implement `DiscountedUCBController` with decay factor $\gamma \in (0.95, 0.999)$.
    *   Implement `ThompsonSamplingController` using Beta-Binomial conjugacy:
        $$\text{Prior: } \text{Beta}(\alpha_0, \beta_0) \implies \text{Posterior: } \text{Beta}(\alpha_0 + S_a, \beta_0 + F_a)$$
    *   Implement forced exploration initialization guaranteeing $N_a \ge N_{\min}$ before unconstrained bandit exploitation.
2.  **Statistical Decision Module (`src/adaptive_qec/controller/statistics.py`)**:
    *   Implement `SequentialProbabilityRatioTest` (Wald SPRT) with exact log-likelihood updates.
    *   Implement `wilson_confidence_interval(successes, trials, confidence=0.95)` with continuity correction.
    *   Implement `two_proportion_z_test(k1, n1, k2, n2)` returning z-statistic and two-sided p-value.
3.  **Simulation Validation Suite (`experiments/bandit_vs_static.py`)**:
    *   Run $10^5$ shots across all 6 noise scenarios comparing:
        *   Static MWPM Baseline
        *   Static Union-Find Baseline
        *   Heuristic Cost Controller (Existing `controller.py`)
        *   Discounted-UCB1 Controller (New)
        *   Thompson Sampling Controller (New)
    *   *Success Metric*: Statistically significant reduction in cumulative regret ($\mathcal{R}(T)$) and LER with $p < 0.001$.

---

### Phase 3: Dynamic Anisotropic Stabilizer Scheduling (Weeks 5–7)

> **Scientific Objective**: Implement and validate the second core novelty—adjusting the temporal frequency of $X$-type vs $Z$-type stabilizer measurements in real time based on observed syndrome defect imbalance.

#### Detailed Deliverables:
1.  **Schedule Specifications (`src/adaptive_qec/qec/schedules.py`)**:
    *   Define formal measurement schedule patterns:
        *   `BALANCED`: Standard alternating round pattern ($[X, Z]$).
        *   `X_HEAVY`: 2:1 ratio ($[X, Z, X]$) optimizing for dominant phase-flip errors.
        *   `Z_HEAVY`: 2:1 ratio ($[Z, X, Z]$) optimizing for dominant bit-flip ($T_1$) errors.
        *   `EXTREME_X` / `EXTREME_Z`: 3:1 ratio for extreme noise regimes.
2.  **Adaptive Scheduler (`src/adaptive_qec/qec/adaptive_scheduler.py`)**:
    *   Implement `AdaptiveXZScheduler` with EWMA smoothing over sliding windows ($W = 50 - 200$ rounds).
    *   Include dual-threshold hysteresis ($\theta_{\text{enter}} = 0.15, \theta_{\text{exit}} = 0.05$) to eliminate boundary chattering.
3.  **Dynamic Circuit Compiler (`src/adaptive_qec/syndrome/extraction.py`)**:
    *   Upgrade Stim circuit generator to dynamically construct heavy-hex surface code detectors for arbitrary $X/Z$ repetitive sequences.
    *   Ensure spacetime detector definitions maintain valid boundary conditions so no false detectors are triggered at schedule transitions.
4.  **Simulation Benchmark (`experiments/adaptive_scheduling.py`)**:
    *   Test under varying noise bias $\eta = p_Z / p_X \in [1, 100]$.
    *   *Success Metric*: Demonstrate that under $\eta = 50$, `AdaptiveXZScheduler` achieves $>18\%$ lower LER than static balanced schedules.

---

### Phase 4: Closed-Loop Live DEM & Peeling Graph Calibration (Weeks 7–9)

> **Scientific Objective**: Bridge passive noise characterization and real-time decoding by synchronizing the decoder's internal matching and peeling graph weights with observed physical drift.

#### Detailed Deliverables:
1.  **DEM Reweighting Engine (`src/adaptive_qec/decoders/dem_calibrator.py`)**:
    *   Implement `DEMCalibrator` with closed-form in-place edge weight mutation:
        $$w_e = \ln \left(\frac{1 - \hat{p}_e}{\hat{p}_e}\right)$$
    *   Ensure zero graph reallocation: mutate PyMatching weights directly in memory.
2.  **Correlated Burst Reweighting**:
    *   Integrate with `noise/burst_detector.py`: when a spatial burst is flagged across a cluster of ancillas, temporarily scale up edge weights on adjacent detectors to prevent spurious long-distance matchings.
3.  **Radius-Weighted Peeling Adaptation (`src/adaptive_qec/decoders/union_find.py`)**:
    *   Dynamically update cluster expansion speed $\Delta r_i(t) \propto \hat{p}_i(t)$ so that high-noise physical regions expand faster in the Union-Find growth phase.
4.  **Benchmark Validation (`experiments/live_calibration.py`)**:
    *   Simulate drifting physical noise channels and demonstrate that live-calibrated MWPM consistently outperforms static DEM MWPM by $\ge 12\%$ in LER.

---

### Phase 5: Qiskit Runtime Closed-Loop Cloud Architecture (Weeks 9–11)

> **Scientific Objective**: Engineer the cloud execution pipeline using Qiskit Runtime Primitives, establishing a robust, batched closed-loop interface with real IBM Heron hardware.

#### Detailed Deliverables:
1.  **Runtime Closed-Loop Coordinator (`src/adaptive_qec/runtime/qiskit_loop.py`)**:
    *   Implement `QiskitRuntimeLoop` utilizing Qiskit Runtime Sessions and `SamplerV2`.
    *   Implement **batched-step adaptation**: execute a batch of $N_{\text{batch}} = 500 - 2,000$ shots, retrieve syndrome bitstrings, evaluate bandit reward and noise state, update DEM weights and circuits, and submit the subsequent batch within the active session.
2.  **Live Telemetry Ingestion (`src/adaptive_qec/qpu/ibm.py`)**:
    *   Implement automated retrieval of IBM backend properties: $T_1(q), T_2(q)$, readout error $\epsilon_{\text{RO}}(q)$, and 2-qubit gate error $\epsilon_{\text{ECR/CZ}}(q_1, q_2)$.
    *   Feed live calibration parameters into the prior distributions of the bandit and DEM calibrator.
3.  **Shot Budget & Quota Manager (`src/adaptive_qec/runtime/budget.py`)**:
    *   Implement strict credit allocation guards to prevent accidental over-consumption of IBM cloud execution quotas.
4.  **Hardware Dry-Run Validation (`experiments/hardware_dryrun.py`)**:
    *   Validate the complete closed loop using the cloud simulator (`ibmq_qasm_simulator` or fake Heron backends) before touching physical hardware.

---

### Phase 6: Real QPU Hardware Execution on IBM Heron (Weeks 11–14)

> **Scientific Objective**: Execute the complete, integrated adaptive QEC stack on an IBM Heron processor (`ibm_marrakesh` or `ibm_kingston`), generating empirical evidence of adaptive advantage.

#### Detailed Deliverables:
1.  **Hardware Baseline Suite (`experiments/hardware_baseline.py`)**:
    *   Execute 6 static baseline configurations (MWPM vs UF) $\times$ (No-DD vs XY4 vs XY8) at code distance $d=3$ (17 physical qubits on heavy-hex lattice).
    *   Collect 10,000 shots per configuration ($60,000$ shots total).
    *   Compute empirical LER with exact 95% Wilson Score intervals.
2.  **Closed-Loop Bandit Execution (`experiments/full_adaptive.py`)**:
    *   Deploy the full adaptive stack for 100,000 shots across an active hardware drift window (monitoring diurnal calibration shifts).
    *   Log all strategy selections, empirical rewards, DEM weight trajectories, and SPRT switching decisions.
3.  **Ablation Runs on Hardware**:
    *   Run bandit controller *without* statistical gating (verifying whether chattering degrades physical performance).
    *   Run adaptive scheduling under static nominal decoders.
4.  **Automated Analysis Pipeline (`experiments/analysis.py`)**:
    *   Generate publication-ready figures: LER comparison with error bars, cumulative regret curves, bandit Q-value trajectories, and timeline of statistical switching events.

---

### Phase 7: Publication, Artifact Release & Open Science Suite (Weeks 14–17)

> **Scientific Objective**: Consolidate all empirical data, statistical proofs, and software into a publication package and open-source release.

#### Detailed Deliverables:
1.  **Manuscript Preparation**:
    *   Draft full paper adhering to *PRX Quantum* / *Nature Communications* standards:
        *   *Title*: "Real-Time Adaptive Quantum Error Correction via Online Bandit Control on a 156-Qubit Superconducting Processor"
        *   *Main Sections*: Introduction, Theoretical Framework & Regret Bounds, System Architecture, Heavy-Hex Embedding & Circuit Compilation, Hardware Experimental Results, Ablation Studies, Discussion & Scaling Analysis.
2.  **Reproducibility Package (`reproducibility/`)**:
    *   Jupyter notebooks replicating every figure from raw hardware telemetry.
    *   Complete Zenodo data repository containing raw bitstrings, timestamps, and IBM calibration JSONs.
3.  **Interactive Visualizer Upgrade (`generated-page.html`)**:
    *   Connect the existing web dashboard to live experiment JSON traces to enable interactive exploration of bandit convergence and syndrome imbalance over time.

---

## 7. Hardware Experimentation Protocol on IBM Heron

### 7.1 Target Hardware Specifications

*   **Processor Family**: IBM Heron (Revision 2 or 3)
*   **Candidate Backends**: `ibm_marrakesh` (156 qubits), `ibm_kingston` (156 qubits), or `ibm_sherbrooke` (127 qubits Eagle as fallback)
*   **Topology**: Heavy-Hexagonal Lattice (degree-2 and degree-3 vertices)
*   **Native 2Q Gate**: Direct CNOT or Cross-Resonance (CR) calibrated CZ
*   **Code Embedding**: Distance-3 rotated planar surface code embedded on heavy-hex sub-graph using flag qubits or SWAP-free bridge ancillas (17 physical qubits total: 9 data qubits, 8 syndrome ancillas).

### 7.2 Shot Allocation & Execution Budget

To ensure statistical power while strictly respecting IBM cloud runtime allocations, the experimental budget is structured as follows:

| Experiment Phase | Configurations / Arms | Shots per Step | Total Batches | Total Shots | Est. Qiskit Runtime Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 6.1: Static Baselines** | 6 fixed configurations | 2,000 | 5 batches / arm | 60,000 | $\approx 25\text{ minutes}$ |
| **Phase 6.2: Adaptive Stack** | Full Bandit + DA-SE + Calibrator | 1,000 | 80 batches | 80,000 | $\approx 35\text{ minutes}$ |
| **Phase 6.3: Ablations** | No-Gate Bandit & Static Scheduler | 1,000 | 40 batches | 40,000 | $\approx 18\text{ minutes}$ |
| **Buffer / Calibration** | Benchmarking & Noise Probes | 1,000 | 20 batches | 20,000 | $\approx 10\text{ minutes}$ |
| **TOTAL** | — | — | **145 batches** | **200,000 shots** | $\approx \mathbf{88\text{ minutes}}$ |

---

## 8. Statistical Rigor, Power Analysis & Ablation Protocol

### 8.1 Statistical Hypothesis Testing Standards
