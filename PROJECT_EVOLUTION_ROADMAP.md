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
