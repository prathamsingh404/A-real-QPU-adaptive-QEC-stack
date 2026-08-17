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

---

## 2. Taxonomy of Failure Modes

| Failure Mode | Physical Mechanism | Observable Signature | Controller Failure Under Naive Policy | CG-CCADS Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **FM-1: High-Frequency Jitter** | Noise fluctuates on timescale $\tau_{\text{drift}} \ll \tau_{\text{adapt}}$ ($\rho \gg 1$) | Syndrome density fluctuates rapidly with low autocorrelation | Constant thrashing between decoder configurations; severe latency lag | **Abstains (`STAY`)**: Timescale ratio $\rho > 1$ drops adaptation confidence to zero. |
| **FM-2: Low-Contrast Drift** | Noise shifts slightly ($p_{x}: 0.010 \to 0.012$), gain is smaller than switching cost | Barely detectable syndrome rate shift; high estimator variance | Thrashing for marginal theoretical gains that are completely destroyed by switching costs | **Dead Zone Filter**: $\Delta \hat{L} < \epsilon_{\text{deadzone}}$ triggers automatic abstention. |
| **FM-3: The Switching Trap** | Noise jumps back and forth between two distinct states with period $\approx 2 \tau_{\text{switch}}$ | Alternating syndrome bias across windows | Endless ping-ponging, suffering repeated $C_{\text{switch}}$ penalties on every transition | **Dual-Threshold Hysteresis**: Separate activation ($\tau_{\text{act}}$) and deactivation ($\tau_{\text{deact}}$) thresholds prevent limit cycles. |
| **FM-4: Deceptive Observable Shift** | Syndrome rates shift due to benign circuit variation without logical error rate shift | Significant syndrome rate jump ($\Delta r > 0$), but counterfactual $\Delta L \approx 0$ | Triggering unnecessary decoder rebuilds without logical improvement | **Counterfactual Disagreement Gating (Gate 2)**: If candidate decoder agrees on $\ge 98\%$ of syndromes, switch is blocked. |
| **FM-5: Incomplete Observability (Hidden Transitions)** | Noise shifts along unobservable stabilizer axes or within short bursts | Delayed syndrome accumulation; low SNR | Belated switching after errors have already compromised logical state | **Multi-Timescale Estimator & Conformal LCB**: Fast window detects change; slow window confirms confidence before commitment. |
| **FM-6: Physical DD Pulse Overhead** | Inserting DD sequences on idle qubits increases gate count and pulse infidelity | Increased logical error rate despite nominal $T_2$ decoupling | Uniform application of DD degrades codes when $T_1/T_2$ are not the dominant bottleneck | **Crossover Threshold Gate**: DD is disabled unless calibrated coherence gap exceeds pulse infidelity threshold. |

---

## 3. Quantitative Boundary Conditions
