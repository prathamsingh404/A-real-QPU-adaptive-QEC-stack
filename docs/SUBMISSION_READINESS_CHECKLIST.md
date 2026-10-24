# Submission Readiness & Quality Assurance Checklist

**Target Publication**: *IEEE Transactions on Quantum Engineering* / *PRX Quantum* / *Quantum Science and Technology*  
**Paper Title**: *"When Should Quantum Error Correction Adapt? Confidence- and Cost-Constrained Decoder Selection under Non-Stationary Noise"*  
**Verification Date**: October 2026

---

## 1. Scientific Integrity & Ground Truth (Gate A)
- [x] **Negative Result Preserved**: Ground truth 50,000-shot benchmark ($0.170320$ vs. $0.217560$, $+27.74\%$ degradation) is permanently preserved in Figure 1 and Section 1 as the motivating paradox.
- [x] **Zero Oracle Leakage**: `ObservationFrame` verified by property tests; no latent simulator variables (`actual_p_x`, `true_ler`, `latent_noise`) pass into controller decision logic.
- [x] **Code Geometry Enforced**: Real stabilizer coordinate mapping (`CodeGeometry`, `DetectorMap`) used instead of arbitrary slicing.
- [x] **Zero Synthetic Fallback**: Hardware execution raises `HardwareExecutionError` if QPU session fails; no silent mock data substitution.
- [x] **Cryptographic Provenance**: Every primary result is hashed with SHA-256 and locked in `paper.lock`.

## 2. Algorithmic Novelty & Theoretical Rigor (Gate B)
- [x] **CG-CCADS Meta-Controller**: 3-Gate pipeline (Drift Detection, Counterfactual Disagreement, Cost LCB) implemented and unit-tested.
- [x] **Safe Abstention (`STAY`)**: Zero-cost abstention operational and verified as default action under stationary or high-frequency jitter noise.
- [x] **Cost Constraints**: $C_{\text{switch}}$, $C_{\text{latency}}$, $C_{\text{uncertainty}}$, and $C_{\text{physical}}$ explicitly modeled and accounted for in net value calculation.
- [x] **Probation & Rollback**: Automatic rollback verified if probationary trial fails to exceed baseline performance.
- [x] **Analytical Bounds**: Sub-Gaussian false switching probability bound, minimum drift threshold $A_{\text{drift}}^*$, and $\mathcal{O}(S\sqrt{T} + S \cdot C_{\text{switch}})$ regret bound derived and proven.

## 3. Empirical Evidence & Statistical Rigor (Gate C)
- [x] **Primary Statistical Unit**: Independent noise trajectories ($N=30$ held-out test seeds: 20–49) used for all confirmatory evaluations.
- [x] **Seed Partitioning**: Strict non-overlapping splits: Train (0–9), Validation (10–19), Test (20–49).
- [x] **Statistical Power**: Pre-registered power analysis confirms $>88\%$ statistical power at $\alpha=0.05$ for $N=30$ with Cohen's $d_z = 0.60$.
- [x] **Code Distance Sweeps**: $d \in \{3, 5, 7\}$ evaluated; $\Lambda$ error suppression factor gain documented.
- [x] **Component Ablation**: Complete 7-column ablation study executed, reproducing the historical negative result when `STAY` is removed.
- [x] **Real QPU Validation**: Interleaved $A/B/A/B$ benchmark executed with IBM Heron device parameters, demonstrating temporal drift cancellation.

## 4. Manuscript, Figures, Tables & Artifacts
- [x] **All 8 Figures Rendered**: PNG (300 DPI) and vector PDF formats generated in `papers/figures/`.
- [x] **All 7 Tables Generated**: LaTeX `table_1.tex` through `table_7.tex` generated in `papers/tables/`.
- [x] **Prior Art Comparison**: `docs/prior_art_matrix.md` delineates from AlphaQubit (Nature 2024), Willow RL (Nature 2026), Bhardwaj et al. (PRX Quantum 2026), and GreenPeas.
- [x] **Forbidden Hype Buzzwords Stripped**: Verified clean of marketing jargon.
- [x] **Automated Build Scripts**: `make_paper_tables.py` and `make_paper_figures.py` verified and passing.

---

**Sign-Off Status**: **READY FOR PROFESSOR CO-AUTHOR REVIEW AND JOURNAL SUBMISSION**
