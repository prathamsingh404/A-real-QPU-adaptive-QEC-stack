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
