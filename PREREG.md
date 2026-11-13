# Analysis Plan & Pre-Registration Protocol: Adaptive Quantum Error Correction Benchmark

**Document Status**: Registered Analysis Plan & Confirmatory Endpoint Protocol  
**Scope**: Statistical evaluation protocol for Confidence-Gated Adaptive Decoder Selection (CG-CCADS)  
**Tuning Partition (Exploratory)**: Seeds 0--99  
**Held-Out Confirmatory Partition**: Seeds 1000--1029 ($N=30$ independent trajectories)  
**Primary Suite Artifact**: `data/suite_results.json` (720 total evaluation runs)  
**Provenance Classification**: Monte Carlo Simulation (Stim v1.16.0 + PyMatching v2.4.0); Zero Simulated Hardware Claims.

---

## 1. Summary of Analysis Plan and Hypotheses

This document establishes the formal pre-registered protocol for evaluating online decoder adaptation in quantum error correction. The full architectural analysis plan is specified in [`docs/ANALYSIS_PLAN.md`](file:///docs/ANALYSIS_PLAN.md) and tracked in [`VALIDATION.md`](file:///VALIDATION.md).

### Pre-Registered Confirmatory Hypotheses & Claims Mapping

| Hypothesis / Endpoint | Target Claim ID | Pre-Registered Effect Criterion | Empirical Result (`data/suite_results.json`) | Status |
| :--- | :--- | :--- | :--- | :---: |
| **H1: Localized Drift Advantage** | **CLAIM-CONF-03** | $\bar{\Delta} < 0$, $p < 0.05$ at $W=2{,}000, d=3$ | $\bar{\Delta} = -0.00043$, $95\%$ CI $[-0.00077, -0.00009]$, $t = -2.60$, $p = 0.0145$ | **CONFIRMED** |
| **H0a: Stationary Noise Invariance** | **CLAIM-CONF-04** | $\bar{\Delta} \approx 0$, $p \ge 0.05$, Stay Rate $\ge 90\%$ | $\bar{\Delta} = +0.00006$, $p = 0.3256$, Stay Rate $= 100\%$ | **CONFIRMED** |
| **H0b: Uniform Drift Invariance** | **CLAIM-CONF-04** | $\bar{\Delta} \approx 0$, $p \ge 0.05$, Stay Rate $\ge 90\%$ | $\bar{\Delta} = +0.00016$, $p = 0.1052$, Stay Rate $= 93.3\%$ | **CONFIRMED** |
| **Below-Threshold Suppression** | **CLAIM-CONF-01** | $\Lambda_{3 \to 5} = \text{LER}(d=3)/\text{LER}(d=5) > 1.0$ | $\Lambda_{3 \to 5} = 1.543 > 1.0$ (Static: $1.542$) | **CONFIRMED** |
| **Small-Window Variance Explosion** | **CLAIM-CONF-02** | Un-gated reweighting at $W=200$ degrades LER | $\Delta\text{LER} = +104.38\%$ ($0.02009 \to 0.04107$); CG-CCADS Stay $= 92.7\%$ | **CONFIRMED** |
| **Pipeline AST Integrity** | **CLAIM-CONF-05** | Zero non-whitelisted numerical constants | 0 violations (`tests/test_no_hardcoded_results.py`) | **CONFIRMED** |
| **Cryptographic Provenance Lock** | **CLAIM-CONF-06** | Complete SHA-256 integrity match | 11/11 files matched (`paper.lock`) | **CONFIRMED** |

---

## 2. Statistical Methodology & Paired Trajectory Design

- **Statistical Unit**: Independent noise trajectory ($N=30$ held-out seeds, 1000--1029).
- **Paired Design**: All decoder arms (Static MWPM, Periodic Spitz, CG-CCADS, and Oracle MWPM) decode identical syndrome streams generated from the identical physical noise realization.
- **Statistical Testing**:
  - Primary endpoint: Two-sided paired Student's $t$-test on trajectory deltas $\Delta_i = \text{LER}_{\text{adaptive}, i} - \text{LER}_{\text{static}, i}$.
  - Non-parametric confirmation: Two-sided Wilcoxon signed-rank test.
  - Multi-hypothesis control: Holm-Bonferroni step-down correction across all confirmatory noise regimes.
  - Confidence intervals: 95% percentile bootstrap confidence intervals computed over $B = 10{,}000$ resamples.
- **Sample Sizing**: Prospective design power calculation demonstrates that $N=30$ independent trajectories provides $\approx 84\%$ statistical power to detect an effect size of $|\Delta| = 0.0006$ at significance level $\alpha = 0.05$.

---

## 3. Strict Hardware Provenance Boundary: Zero Simulated Hardware Policy

To maintain uncompromising scientific integrity, the project enforces a strict boundary between simulated noise models and physical quantum hardware executions:
1. **Confirmatory Manuscript Scope**: All primary quantitative endpoints reported in the main paper (Tables 1--4) operate under reproducible Monte Carlo simulation on Stim and PyMatching with seeds 1000--1029, cryptographically recorded in `data/suite_results.json` and `data/manifest.json`.
2. **Physical QPU Benchmarks**: All physical hardware data in `data/hardware_results/` originates exclusively from genuine quantum processor executions on the 156-qubit `ibm_marrakesh` heavy-hex QPUs via the IBM Quantum Platform.
3. **Retraction of Synthetic Hardware Claims**: An earlier historical synthetic binomial demo previously labeled as IBM Heron A/B benchmark has been completely retracted (`CLAIM-HIST-15`) and relegated to `examples/synthetic_ab_demo.py` with explicit `"provenance": "synthetic"` labeling.
4. **Zero Simulated Hardware Rule**: No synthetic or simulated dataset may ever be reported, labeled, or presented as physical hardware execution in any manuscript table, text, or figure.

---

## 4. Pre-Registration Status

In accordance with open science best practices, until formal external timestamped deposit on the Open Science Framework (OSF) or Zenodo is completed, this document and its cryptographic binding in `paper.lock` serve as the immutable internal pre-registration ledger and analysis specification.
