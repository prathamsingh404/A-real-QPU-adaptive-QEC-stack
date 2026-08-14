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

---

## 2. Statistical Unit & Primary Endpoints

- **Unit of Analysis**: The statistical unit is the **independent stochastic noise trajectory** ($N=30$ seeds), evaluated across all windows $W$ under identical sampled syndrome streams (paired design).
- **Primary Endpoint**: Paired trajectory-level delta:
  $$\Delta_i = \text{LER}_{\text{adaptive}, i} - \text{LER}_{\text{static}, i}$$
- **Primary Statistical Test**:
  - Paired Student's $t$-test (with two-sided alternative) across trajectory deltas.
  - Paired Percentile Bootstrap (10,000 resamples) for $95\%$ Confidence Intervals of mean delta $\bar{\Delta}$.
  - Holm-Bonferroni step-down correction for multiple testing across the confirmatory regimes (family-wise error rate $\alpha = 0.05$).

---

## 3. Data Partitions & Protocol Partitioning

To strictly prevent p-hacking and post-hoc parameter overfitting:
1. **Tuning Partition (Exploratory / Calibration)**:
   - Random Seeds: `0` to `99`.
   - Used to calibrate detector defect reference rates ($\lambda_{\text{ref}}$), CUSUM drift detection threshold ($T_{\text{drift}} = 3.0$), and the LCB update deadzone ($\epsilon_{\text{deadzone}} = 0.005$).
2. **Confirmatory Held-Out Partition**:
   - Random Seeds: `1000` to `1029` ($N=30$ independent trajectories).
   - Evaluated exactly once under frozen parameters and configurations in `configs/suite.yaml`.
   - All results tables and figures in the manuscript are compiled exclusively from this partition.
