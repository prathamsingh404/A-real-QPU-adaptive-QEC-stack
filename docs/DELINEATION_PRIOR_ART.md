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
