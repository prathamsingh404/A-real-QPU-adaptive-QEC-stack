# Submission Readiness & Quality Assurance Checklist

**Target Publication**: *IEEE Transactions on Quantum Engineering* / *Quantum Science and Technology*  
**Paper Title**: *"When Does Online Decoder Reweighting Pay Off in Quantum Error Correction? A Confidence-Gated Framework"*  
**Verification Date**: October 2026 (Integrity Reset & Rigorous Pipeline Compilation)

---

## 1. Scientific Integrity & Observation Boundaries (Gate A)
- [x] **Finite-Sample Variance Identified**: Motivating tradeoff established: at $W=200$, naive reweighting increases logical error by $+104.38\%$ due to binomial covariance noise, while localized drift at $W \ge 2{,}000$ yields up to $29.5\%$ theoretical gain.
- [x] **Zero Oracle Leakage**: The controller ingests strictly label-free syndrome arrays ($X \in \{0, 1\}^{W \times |V|}$); no ground-truth logical observable bits or simulator parameters pass into decision logic.
- [x] **Realistic Noise Boundary**: Evaluated strictly below threshold ($p=0.004 < p_{\text{th}} \approx 0.01$), where increasing code distance from $d=3$ to $d=5$ exponentially suppresses logical error ($\Lambda_{3 \to 5} = 1.543 > 1.0$).
- [x] **Hardware Provenance Delineated**: Exploratory runs on IBM Quantum (ibm_marrakesh) documented honestly; synthetic A/B demonstrations isolated to `examples/synthetic_ab_demo.py` with explicit `"provenance": "synthetic"` metadata.
- [x] **Cryptographic Provenance Lock**: All empirical data, tables, and figures hashed with SHA-256 in `paper.lock` and verified in continuous integration.

## 2. Algorithmic Novelty & Theoretical Rigor (Gate B)
- [x] **CG-CCADS Meta-Controller**: 3-Gate pipeline (Defect CUSUM, Bonferroni FWER LCB on Spitz edge shifts, and Cost-Constrained Threshold) implemented and unit-tested.
- [x] **Safe Abstention (`STAY`)**: Zero-cost abstention operational and verified as the default action under stationary noise ($92.7\%$ abstention rate across confirmatory suite).
- [x] **Cost Constraints**: Physical switching penalty $C_{\text{switch}}$ and latency tax $C_{\text{latency}}$ explicitly accounted for in decision barriers.
- [x] **Mathematical Formulations**: Exact single-fault covariance-parity estimators (Spitz et al. 2018), FWER concentration bounds, Page CUSUM Average Run Length ($\text{ARL}_0$), and piecewise-stationary regret bounds documented in `docs/THEORETICAL_PROOFS_AND_BOUNDS.md`.

## 3. Empirical Evidence & Statistical Rigor (Gate C)
- [x] **Primary Statistical Unit**: Independent stochastic noise trajectories ($N=30$ held-out test seeds: 1000–1029) evaluated across identical syndrome streams (paired design).
- [x] **Seed Partitioning**: Strict non-overlapping splits: Tuning / Exploratory (seeds 0–99) vs. Confirmatory Held-Out (seeds 1000–1029).
- [x] **Sample Sizing**: Prospective design power calculation confirms $N=30$ trajectories provide $\sim 84\%$ power to detect a relevant effect size of $|\Delta| = 0.0006$ at $\alpha = 0.05$.
- [x] **Below-Threshold Distance Scaling**: Rotated surface codes evaluated at $d \in \{3, 5\}$, achieving $\Lambda = 1.543 > 1.0$.
- [x] **Component Ablation**: Full ablation across window sizes $W \in \{200, 1000, 2000\}$ demonstrating that CG-CCADS safely avoids the $+104.38\%$ variance penalty of unconstrained adaptation.

## 4. Manuscript, Figures, Tables & Artifacts
- [x] **All Figures Rendered**: PNG (300 DPI) and vector PDF formats dynamically generated in `papers/figures/` from `data/suite_results.json`.
- [x] **All Tables Generated**: LaTeX `table_1.tex` through `table_4.tex` and `table_7.tex` dynamically compiled in `papers/tables/`.
- [x] **Zero Hardcoded Result Literals**: Verified by Abstract Syntax Tree (AST) continuous integration test (`tests/test_no_hardcoded_results.py`).
- [x] **Prior Art Comparison**: Explicitly positioned against Spitz et al. (2018), Bhardwaj et al. (PRX Quantum 2026), AlphaQubit (Nature 2024), and Google Willow RL (Nature 2026).
- [x] **AI Assistance Disclosed**: Full disclosure statement provided in manuscript and README in adherence to IEEE author policies.
