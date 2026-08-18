# Comparative Prior Art Matrix & Literature Grounding

**Manuscript Reference Document**  
**Associated Target Publication**: *IEEE Transactions on Quantum Engineering* / *Quantum Science and Technology*  
**Paper Title**: *"When Does Online Decoder Reweighting Pay Off in Quantum Error Correction? A Confidence-Gated Framework"*

---

## 1. Prior Art Comparison Matrix

| Approach / Work | Primary Target | Adaptation Mechanism | Considers Switching / Latency Cost? | Has Explicit Abstention (`STAY`)? | Observation Boundary Enforced? | Regret / Theoretical Bounds? | Evaluation Modality |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Static MWPM**<br>(Dennis et al. 2002; Fowler et al. 2012) | Stationary Pauli noise | None (fixed factory prior) | N/A | Always stays | Yes (syndromes only) | Static threshold proofs | Multi-platform QPU |
| **Spitz et al.**<br>(*arXiv:1712.02360*, 2018) | Time-dependent noise | Unconditional covariance-parity edge estimator | ❌ No | ❌ No | Yes (label-free syndromes) | Estimator asymptotic variance | Numerical simulation |
| **Bhardwaj et al.**<br>(*PRX Quantum*, 2026) | Drifting error channels | Sliding-window DEM reweighting | ❌ No | ❌ No | Yes (syndromes only) | Asymptotic threshold gain | Numerical simulation |
| **Calibrated Decoders**<br>(Chen et al., *PRL* 2022) | Inhomogeneous device noise | Offline calibrated edge weights | ❌ No (offline) | N/A | Offline calibration | None reported | Superconducting QPU |
| **AlphaQubit**<br>(Google DeepMind, *Nature* 2024) | Static processor instance | Deep recurrent transformer neural network | ❌ No | ❌ No | Soft/analog readouts + syndromes | Empirical neural scaling | 72-qubit Sycamore QPU |
| **Willow RL Control**<br>(Google Quantum AI, *Nature* 2026) | Physical Hamiltonian drift | Reinforcement learning pulse/frequency tuning | ⚠️ Partial (loss reward) | ❌ No | Calibration + leakage telemetry | Empirical RL convergence | Willow QPU |
| **Local Clustering Decoder**<br>(Riverlane, *arXiv:2411.10343*) | Sub-microsecond real-time decoding | Distributed local cluster growth on FPGA/ASIC | ⚠️ Real-time latency | ❌ No | Local syndrome windows | Hardware execution bounds | FPGA emulation |
| **CG-CCADS (This Work)** | **Non-stationary QEC orchestration** | **3-Gate Confidence & Cost Constrained Selection** | **✅ Explicitly quantified** ($C_{\text{switch}} + C_{\text{latency}}$) | **✅ Core primitive** (Default is zero-cost `STAY`) | **✅ Strict label-free boundaries** (Zero oracle leakage) | **✅ FWER-controlled LCB & dynamic regret bounds** | **Held-out simulation suite & reproducible pipeline** |

---

## 2. Complementary Positioning vs. Frontier Literature

### 2.1 Spitz et al. (arXiv:1712.02360, 2018)
- **Foundational Insight**: Derived the analytical single-fault parity-covariance equations enabling edge error probabilities in a matching graph to be estimated directly from syndrome measurement coincidences without requiring logical state labels.
- **The Open Gap**: Unconditional periodic re-estimation was evaluated without considering finite-sample estimation variance in small sample windows ($W \le 1{,}000$), where coincidence noise severely corrupts Dijkstra graph weights.
- **Our Positioning**: We adopt the Spitz covariance-parity equations as our candidate graph estimator, but embed it within a sequential 3-gate statistical filter that computes Bonferroni-corrected Lower Confidence Bounds (LCB) before accepting any weight recomputation.

### 2.2 Bhardwaj et al. (PRX Quantum 7, 033024, 2026)
- **Foundational Insight**: Demonstrated that updating MWPM edge weights in drifting superconducting channels reduces logical error under slow parameter variations.
- **The Open Gap**: Assumed that sliding-window reweighting is universally beneficial, evaluating regimes where sample sizes were asymptotically favorable. Did not analyze small-window variance explosion, uniform drift invariance, or decision-theoretic abstention.
- **Our Positioning**: We delineate the operational boundary: at $W=200$, naive unconditional reweighting degrades logical error by $+104.38\%$, and uniform global drift provides $<2\%$ headroom. CG-CCADS introduces the confidence-gated decision rule that enforces safe abstention (`STAY`) under null drift and small windows, updating only when localized drift provides statistically conclusive advantage.

### 2.3 AlphaQubit (Google DeepMind, *Nature* 2024)
- **Foundational Insight**: Achieved breakthrough decoding accuracy on 72-qubit Sycamore using transformer architectures trained on soft analog readout values and syndrome histories.
- **The Open Gap**: AlphaQubit is an individual decoder primitive. It does not address supervisory orchestration: when should a system incur high-latency neural inference versus remaining with ultra-fast matching or static priors?
- **Our Positioning**: CG-CCADS operates at the supervisory orchestration layer above individual decoding engines, regulating classical update dispatches on macro-timescales ($W$ syndrome rounds).

### 2.4 Google Willow RL Control (*Nature* 2026)
- **Foundational Insight**: Uses reinforcement learning to optimize microwave pulse parameters, frequencies, and physical gate calibrations at the physical device layer.
- **Our Positioning**: Google Willow intervenes on the **physical quantum Hamiltonian** (pulse-level control). CG-CCADS operates on the **classical information layer** (deciding how syndrome graphs are weighted), avoiding physical state perturbation.
