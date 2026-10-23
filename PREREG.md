# Analysis Plan: Adaptive Quantum Error Correction Benchmark

**Document Status**: Analysis Plan & Protocol Specification (Refer to `docs/ANALYSIS_PLAN.md`)  
**Scope**: Statistical evaluation protocol for Confidence-Gated Adaptive Decoder Selection (CG-CCADS)  
**Tuning Partition**: Seeds 0--99  
**Held-Out Confirmatory Partition**: Seeds 1000--1029 ($N=30$ independent trajectories)  

---

## 1. Summary of Analysis Plan

This document establishes the experimental protocol for evaluating online decoder adaptation in quantum error correction. The full analysis plan is detailed in [`docs/ANALYSIS_PLAN.md`](file:///docs/ANALYSIS_PLAN.md).

### Core Hypotheses
- **H1 (Localized Drift Advantage)**: Under localized physical drift (e.g., hotspot qubit), online covariance-parity reweighting with confidence gating outperforms static factory-calibrated MWPM ($\bar{\Delta} < 0, p < 0.05$).
- **H0 (Safe Abstention Under Negative Controls)**: Under stationary noise, uniform global drift, or inadequate window sizes ($W \le 1{,}000$), the controller must safely abstain (`STAY`), avoiding false adaptation and finite-sample variance explosion.

### Statistical Unit & Methodology
- **Unit**: Independent noise trajectory ($N=30$ seeds).
- **Design**: Paired evaluation where all decoder arms decode identical syndrome streams.
- **Statistical Testing**: Two-sided paired Student's $t$-test / Wilcoxon signed-rank test, $95\%$ percentile bootstrap confidence intervals, and Holm-Bonferroni step-down correction across confirmatory regimes.
- **Sample Sizing**: Prospective design power calculation indicates $N=30$ provides $\sim 84\%$ power to detect an effect size of $|\Delta| = 0.0006$ at $\alpha = 0.05$. (This is a sample-size planning calculation, not empirical evidence.)

### Pre-Registration Status
In accordance with open science practices, until external timestamped deposit on OSF or Zenodo is completed, this protocol serves as the internal preregistration and analysis specification.
