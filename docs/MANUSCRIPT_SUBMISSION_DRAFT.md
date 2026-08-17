# Manuscript Draft: When Does Online Decoder Reweighting Pay Off in Quantum Error Correction? A Confidence-Gated Framework

**Target Submission**: *IEEE Transactions on Quantum Engineering*  
**Authors**: Pratham Singh  
**Artifact Lock**: `paper.lock` (SHA-256 Cryptographically Bound)  
**Code Repository**: Open Source Reproducible Suite  

---

## Abstract

Physical quantum error correction (QEC) processors operate in non-stationary noise environments driven by two-level system (TLS) defect dynamics, $1/f$ flux drift, and cryogenic temperature variations. While adaptive decoding via online syndrome-based noise re-estimation is intuitively appealing, we identify a fundamental operational tension: **naive online reweighting under small observation windows ($W \le 1{,}000$) severely inflates the logical error rate** (e.g., $+104.4\%$ relative logical error penalty at $W=200$, rising from $0.02009$ to $0.04107$) due to binomial sample variance in coincidence-parity estimators. Conversely, under localized spatial drift (e.g., a drifting hotspot qubit), reweighting at large window sizes ($W \ge 2{,}000$) captures the true error channel and yields up to $29.5\%$ theoretical logical error reduction.

To resolve this trade-off, we introduce **Confidence-Gated Cost-Constrained Adaptive Decoder Selection (CG-CCADS)**, an online meta-control framework operating strictly on label-free syndrome statistics. CG-CCADS decouples monitoring from updating via three sequential statistical gates: (1) detector defect rate CUSUM drift detection, (2) family-wise error rate (FWER) controlled lower confidence bounds (LCB) on edge covariance shifts via the Spitz et al. parity estimator, and (3) an update-cost deadzone preventing futile graph recomputations. Under stationary baseline noise and uniform drift, the controller safely defaults to zero-cost abstention (`STAY`), suppressing variance explosion with a $92.7\%$ abstention rate.

We evaluate the framework across 30 independent held-out pseudo-random test trajectories (seeds 1000--1029) on rotated surface codes below threshold ($p=0.004 < p_{\text{th}}$, $d \in \{3, 5\}$). Below-threshold distance scaling demonstrates logical suppression $\Lambda_{3 \to 5} = 1.543 > 1.0$. All simulation data, configurations, and figures are bound to an automated, reproducible pipeline verified by an SHA-256 cryptographic provenance lock with zero hardcoded result literals.

---

## 1. Introduction

Fault-tolerant quantum computing architectures rely on Quantum Error Correction (QEC) to protect logical quantum states from environmental decoherence and gate infidelities. Standard decoding algorithms, such as Minimum-Weight Perfect Matching (MWPM) and recent neural decoders like AlphaQubit (*Nature* 2024), map syndrome measurements to error corrections assuming stationary Pauli noise calibrated at device initialization.

However, physical superconducting and trapped-ion processors exhibit time-dependent non-stationary drift. Two-level system (TLS) defect fluctuators, magnetic flux noise, and cryogenic thermal shifts cause localized physical qubit error rates to drift by factors of $2\times$ to $5\times$ over minutes to hours.

In response, several recent works have proposed online adaptive QEC, wherein edge weights in the detector error model (DEM) are continuously re-estimated from running syndrome histories. While continuous adaptation appears beneficial asymptotically, we uncover a crucial operational caveat: **finite-sample estimation variance**.

When observation window sizes are constrained ($W \le 1{,}000$ syndrome rounds), sample noise dominates the detector coincidence statistics, corrupting the decoding graph and substantially increasing the logical error rate. Furthermore, under uniform global drift, scaling all physical error rates preserves the relative Dijkstra shortest paths, providing negligible headroom ($< 2\%$) for online graph reweighting while incurring graph recompilation costs.

In this work, we formalize the conditions under which online decoder reweighting is statistically justified. We design **Confidence-Gated Cost-Constrained Adaptive Decoder Selection (CG-CCADS)**, an orchestration layer that monitors syndrome defect streams label-free and updates the decoder only when localized drift exceeds rigorous confidence bounds.

---

## 2. Core Contributions

1. **Characterization of Finite-Window Reweighting Variance**:
   Demonstrating analytically and empirically that online covariance-parity re-estimation under small windows ($W \le 1{,}000$) severely degrades logical error rate due to binomial estimation variance, while identifying localized drift as the primary regime with genuine adaptation headroom.
2. **Label-Free 3-Gate Statistical Control Architecture**:
   Developing a controller that monitors detector firing CUSUM statistics, evaluates Bonferroni-corrected Lower Confidence Bounds (LCB) on edge deviations, and enforces safe zero-cost abstention (`STAY`) under stationary or low-sample regimes.
3. **Pre-Registered Confirmatory Suite with Cryptographic Provenance Lock**:
   Implementing an automated end-to-end simulation pipeline across 30 held-out trajectories ($N=30$, seeds 1000--1029) on rotated surface codes below threshold ($p=0.004$, $d \in \{3, 5\}$) with zero hardcoded result literals, verified by SHA-256 provenance hashes.
4. **Transparent Delineation of Orchestration Latency and Physical Realities**:
   Profiling host orchestration latency stack with sub-millisecond overhead, and honestly contextualizing software emulation vs. long-duration hardware requirements.

---
