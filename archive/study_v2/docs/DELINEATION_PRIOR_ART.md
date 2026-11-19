# Scientific Delineation from Frontier Prior Art (2022–2026)

**Document Type**: Scientific Architecture & Manuscript Positioning  
**Target Venue**: *IEEE Transactions on Quantum Engineering* / *Quantum Science and Technology*  
**Scope**: Systematic Delineation of Confidence-Gated Cost-Constrained Adaptive Decoder Selection (CG-CCADS) from Classical, Neural, and Hardware QEC Implementations

---

## 1. The Core Scientific Question

A foundational reviewer inquiry in adaptive quantum error correction is:
> *"Online noise estimation and adaptive decoding graphs have already been studied in the literature. What specific problem does this work address that was not previously understood?"*

This document formalizes the explicit mathematical, architectural, and operational boundaries separating **CG-CCADS** from frontier publications.

---

## 2. Comparative Delineation Matrix

| Framework | Primary Contribution | Operational Layer | Considers Switching / Latency Cost? | Has Explicit `STAY` (Abstention)? | Our Key Delineation |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Static MWPM**<br>*(Dennis 2002; Fowler 2012)* | Classical blossom matching on fixed factory graph | Real-time decoder | N/A | Always stays | Static MWPM cannot adapt to drifting parameters. CG-CCADS retains static MWPM as its default zero-cost baseline prior. |
| **Spitz et al.**<br>*(arXiv:1712.02360, 2018)* | Parity-covariance equations for single-fault edge probabilities | Graph weight estimator | ❌ No | ❌ No | Spitz et al. derived the analytical parity estimator. We discover that applying it unconditionally under small windows ($W \le 1{,}000$) severely degrades LER ($+104.4\%$ at $W=200$). We build the gating layer that makes it safe. |
| **Bhardwaj et al.**<br>*(PRX Quantum, 2026)* | Sliding-window dynamic graph reweighting for drifting channels | Adaptive MWPM | ❌ No | ❌ No | Bhardwaj et al. demonstrated asymptotic gains with large sample windows. Our work discovers the finite-sample variance explosion at small windows, proves uniform drift provides $<2\%$ headroom, and provides the 3-gate statistical barrier deciding *when* to adapt. |
| **AlphaQubit**<br>*(Google DeepMind, Nature 2024)* | Transformer neural network predicting corrections from analog readouts | Neural decoder primitive | ❌ No | ❌ No | AlphaQubit is an individual decoder engine. CG-CCADS is a supervisory orchestrator operating on macro-intervals, regulating when graph updates or heavy inference are statistically justified. |
| **Willow RL Control**<br>*(Google Quantum AI, Nature 2026)* | Deep reinforcement learning optimizing physical microwave pulses and frequencies | Physical hardware calibration | ⚠️ Partial (loss reward) | ❌ No | Google Willow tunes the physical Hamiltonian. CG-CCADS operates strictly on the classical information layer (syndrome graph weights), avoiding physical state disruption. |
| **Riverlane LCD**<br>*(arXiv:2411.10343, 2024)* | Sub-microsecond local clustering on FPGA/ASIC | Real-time hardware decoder | ⚠️ Hardware clock cycles | ❌ No | Riverlane LCD focuses on sub-microsecond FPGA hardware decoding. CG-CCADS is an offline / supervisory software orchestration layer running at macro-timescales (every $W$ rounds), complementing rather than replacing FPGA decoders. |

---

## 3. Detailed Scientific Demarcations

### 3.1. Delineation from Bhardwaj et al. (PRX Quantum 7, 033024, 2026)
Bhardwaj et al. demonstrated that dynamically updating Detector Error Model (DEM) weights from running syndrome observations can reduce logical error under slow parameter drift.

However, their study evaluated regimes where sample sizes were asymptotically sufficient and assumed continuous adaptation is universally desirable. In this work, we identify two critical physical regimes where unconditional adaptation fails:
1. **The Finite-Sample Variance Explosion**:
   At small window sizes ($W=200$), empirical covariance estimates are dominated by finite-sample coincidence noise. Corrupted edge weights degrade Dijkstra shortest paths, increasing the logical error rate from $0.02009$ to $0.04107$ ($+104.38\%$ penalty vs. static baseline).
2. **The Uniform Global Drift Invariance**:
   When physical error rates drift uniformly across all qubits and couplers, all DEM edge weights shift by an approximately constant additive factor $\approx \ln(c)$. Because additive shifts preserve Dijkstra shortest paths, online adaptation provides negligible headroom ($< 1.7\%$), while consuming graph recompilation compute cycles.

CG-CCADS introduces the **decision-theoretic gating framework** that detects these regimes label-free and safely abstains (`STAY`), protecting fidelity.

### 3.2. Delineation from Spitz et al. (arXiv:1712.02360, 2018)
Spitz, Varbanov, and Terhal established the theoretical foundation showing that single-fault edge probabilities $p_{uv}$ and boundary probabilities $p_b$ can be reconstructed from 2-point detector coincidences without logical observable labels.

Our work builds directly on the Spitz covariance-parity equations. However, where Spitz et al. treated the estimator as an unconditional analytic solver, we discover its finite-sample vulnerability and formulate it as a hypothesis testing problem. By applying Bonferroni family-wise error rate control:
$$\text{LCB}_{uv} = |\hat{p}_{uv} - p_{uv, \text{ref}}| - z_{1 - \alpha / (2|E|)} \sqrt{\frac{\hat{p}_{uv}(1 - \hat{p}_{uv})}{W}}$$
CG-CCADS converts the Spitz estimator from a brittle empirical formula into a statistically protected, risk-bounded online controller.

### 3.3. Delineation from Real-Time Hardware Decoders (Riverlane LCD / FPGA Decoders)
Real-time decoding during active fault-tolerant computation must execute within the coherence cycle of the physical qubits (sub-microsecond timescales for superconducting qubits). Fast decoders like Riverlane's Local Clustering Decoder (LCD) are implemented in FPGA/ASIC hardware pipelines for this purpose.

We explicitly delineate CG-CCADS from real-time decoders:
- **CG-CCADS is NOT an FPGA microcode replacement for real-time decoding.**
- **CG-CCADS is a supervisory software orchestration layer.** It operates at **macro-intervals** (every $W$ syndrome rounds, corresponding to hundreds of microseconds to milliseconds on physical hardware).
- While the sub-microsecond FPGA decoder processes incoming syndromes round-by-round, the supervisory CG-CCADS layer runs concurrently on the host controller, analyzing accumulated syndrome windows to determine whether graph edge weights should be updated.
