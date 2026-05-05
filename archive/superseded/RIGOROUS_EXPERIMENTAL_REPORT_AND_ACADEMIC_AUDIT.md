# Rigorous Experimental Report and Academic Audit
**Date**: October 2026  
**Subject**: Adaptive QEC Stack - Forensic Audit, Integrity Reset & Research Reform  
**Target Venues**: IEEE Transactions on Quantum Engineering (TQE) / ACM Transactions on Quantum Computing (TQC)

---

## 1. Executive Summary & Audit Resolution

A rigorous audit of the `A-real-QPU-adaptive-QEC-stack` repository was conducted to determine whether its codebase, experimental protocols, and empirical data support a publishable peer-reviewed scientific paper.

### Core Verdict:
The project's central scientific premise—that macro-timescale closed-loop adaptation improves fault-tolerant quantum error correction under non-stationary noise—is **unproven and currently unsupported by the data in this repository**.
- In standard simulation benchmarks ($d=3$, 10,000 shots), the adaptive controller achieves a statistically null result compared to static MWPM ($z = -0.30, p = 0.76 > 0.05$).
- In high-statistics sweeps (50,000 shots), the adaptive controller performed **27.74% worse** than static MWPM ($LER = 0.21756$ vs $0.17032$).
- Prior documentation claiming a "+10.05% empirical breakthrough at $d=5$" was based on an isolated run (`seed=47`). Multi-seed evaluations yield **12% to 14% higher logical error rates** for adaptive switching than static MWPM.
- Crucially, the simulation injected noise via artificial detector bit flips without modifying the underlying logical observable, while the controller peeked directly at ground-truth noise schedules (`schedule.is_leakage_active`, true `p_2q`) and invoked hardcoded fast-path rules.

All inflated claims, unverified superlatives, and disconnected metrics have been cataloged in [VALIDATION.md](VALIDATION.md) and programmatically verified via `scripts/make_claims.py`.

---

## 2. Forensic Analysis of Disproven & Downgraded Claims

### 2.1 The Distance $d=5$ Retraction (Claim 13)
- **Claimed**: $+10.05\%$ error reduction ($z = -7.96, p = 1.6 \times 10^{-15}$) at $d=5$ with sublinear regret $R_T = 0.26$.
- **Audit Findings**:
  1. **Single-Seed Anomaly**: The committed artifact (`adaptive_vs_static_d5_20260930_225600.json`) used `seed=47`. A 6-seed evaluation showed adaptive performance dropping to 12–14% worse than static MWPM.
  2. **Oracle Peeking**: In `adaptive_vs_static.py`, the controller inspected `schedule.is_leakage_active(window_idx)` directly from the generator rather than estimating leakage from syndrome autocorrelation.
  3. **Hardcoded Fast-Paths**: In `controller.py`, a deterministic rule (`if leakage_fraction >= 0.15 and distance >= 5: force UF+XY4`) bypassed all cost optimization and bandit exploration.
  4. **Observable Disconnect**: Synthetic leakage was injected by flipping detector bits 2 and 5 to 1 with 85% probability in the syndrome array, while the logical observable was sampled prior to injection. Consequently, MWPM predicted corrections against uncorrupted logical states, inducing artificial logical errors.
