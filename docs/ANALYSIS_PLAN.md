# Confirmatory Analysis Plan: Adaptive Quantum Error Correction Benchmark

**Status**: Protocol & Analysis Plan  
**Date**: October 2026  
**Scope**: Statistical evaluation protocol for Confidence-Gated Adaptive Decoder Selection (CG-CCADS)  
**Authors**: Pratham Singh  

> **Note on Protocol Status**: This document specifies the analysis plan governing the confirmatory evaluation. In accordance with open science standards, prior to external timestamped deposit (e.g., on OSF or Zenodo), this document serves as the internal preregistration and analysis specification.

---

## 1. Research Hypotheses

### Primary Hypothesis (H1: Localized Drift Payoff)
Under localized, spatial drift (e.g., a localized transmon qubit hotspot where error rate shifts from $0.4\%$ to $5.0\%$), an online adaptive decoder utilizing Spitz et al. covariance-parity re-estimation with confidence gating will achieve a statistically significant reduction in logical error rate (LER) compared to static factory-calibrated MWPM:
$$\text{LER}_{\text{CG-CCADS}} < \text{LER}_{\text{static-MWPM}} \quad (\bar{\Delta} < 0, p < 0.05)$$

### Null Hypothesis / Negative Controls (H0: Stationary & Uniform Drift Abstention)
1. **Stationary Baseline**: Under stationary Pauli noise, decoder re-estimation has no true parameter drift to exploit. The optimal policy is safe abstention (`STAY`).
2. **Uniform Global Drift**: Scaling all physical gate and measurement error rates uniformly alters error graph weights by a constant additive shift, leaving Dijkstra shortest paths predominantly invariant. Adaptation headroom is negligible ($< 2\%$).
3. **Small Window Variance (Finite-Sample Tax)**: For small estimation windows ($W \le 1{,}000$), finite-sample coincidence noise causes naive reweighting to severely degrade logical error rate ($+104.4\%$ at $W=200$). CG-CCADS must safely abstain (`STAY`) in $>90\%$ of decision epochs.
