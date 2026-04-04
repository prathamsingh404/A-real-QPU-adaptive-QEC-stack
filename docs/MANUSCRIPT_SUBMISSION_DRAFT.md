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
