# Scientific Failure Modes & Operational Boundaries of Adaptive QEC

**Manuscript Reference Document**  
**Associated Target Publication**: *IEEE Transactions on Quantum Engineering* / *Quantum Science and Technology*  
**Architecture**: Confidence-Gated Cost-Constrained Adaptive Decoder Selection (CG-CCADS)

---

## 1. Executive Summary & The Boundary Principle

Adaptive Quantum Error Correction (QEC) is not universally beneficial. A central scientific contribution of this work is demystifying the adaptation paradox: **in many physical regimes, adapting worsens the logical error rate**.

Continuous or unconstrained adaptation incurs four distinct penalties:
1. **Switching Penalty ($C_{\text{switch}}$)**: Code reconstruction overhead, graph lookup invalidation, and pipeline flushes.
2. **Latency Penalty ($C_{\text{latency}}$)**: Delay in applying corrections while evaluating alternative decoders.
3. **Estimation Variance Penalty ($C_{\text{uncertainty}}$)**: Misadaptation triggered by sample variance in short syndrome windows.
4. **Physical Damaging Interventions ($C_{\text{physical}}$)**: Dynamic decoupling (DD) pulse errors or gate miscalibrations exceeding dephasing mitigation benefit.

Adaptive QEC is beneficial **if and only if**:
$$\text{LCB}(\Delta L_t) = \hat{\Delta L}_t - z_{1-\alpha}\sigma_{\Delta} > C_{\text{switch}} + C_{\text{latency}} + C_{\text{compute}} + C_{\text{physical}}$$

Below, we detail the six primary failure modes and their exact operational boundaries.
