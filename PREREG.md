# Pre-Registration Plan: Adaptive Quantum Error Correction Benchmark (PREREG.md)

**Document Status**: PRE-REGISTERED PROTOCOL  
**Date**: October 2026  
**Registration Target**: OSF / Zenodo Pre-Registration Archive  
**Primary Investigators**: AdaptiveQEC Research Group  

---

## 1. Research Hypotheses

### Primary Hypothesis (H1)
Under physically realistic non-stationary noise (low-frequency $1/f$ drift in gate error rates and idle dephasing, localized cosmic-ray-like burst avalanches, and stochastic transmon leakage), an adaptive controller that dynamically re-estimates detector error models (DEM) and selects error mitigation strategies achieves a lower logical error rate (LER) than a static minimum-weight perfect matching (MWPM) baseline using factory calibration:
$$\text{LER}_{\text{adaptive}} < \text{LER}_{\text{static-MWPM}}$$

### Null Hypothesis (H0)
$$\text{LER}_{\text{adaptive}} \ge \text{LER}_{\text{static-MWPM}}$$
The overhead of syndrome-based estimation error, misclassification, and frequent switching penalties offsets any theoretical advantage of adaptation.

### Secondary Hypotheses
- **H2 (Baseline Ladder Rank)**: Sliding-window DEM re-estimation (Bhardwaj et al. style) and graph re-weighting (DGR style) outperform static factory DEMs under monotonic gate drift.
- **H3 (Regime Boundary)**: Adaptive selection yields a positive effect size only when drift timescales $T_{\text{drift}}$ are significantly longer than the estimation window latency ($T_{\text{drift}} \gg W \cdot N_{\text{shots}}$) and drift amplitudes exceed physical threshold margins ($\Delta p / p > 0.5$).

---

## 2. Primary Metrics & Analysis Unit

- **Unit of Analysis**: The unit of statistical analysis is an **independent noise trajectory** (a complete run across $W$ windows under an independent stochastic seed), NOT an individual shot.
- **Primary Metric**: Paired difference in mean logical error rate per trajectory:
  $$\Delta_i = \text{LER}_{\text{static}, i} - \text{LER}_{\text{adaptive}, i}$$
- **Statistical Tests**:
  - Two-sided Paired Wilcoxon Signed-Rank Test across trajectories.
  - Paired Percentile Bootstrap (10,000 resamples) for 95% Confidence Intervals of effect size.
  - Holm-Bonferroni correction for multiple hypothesis comparisons across code distances $d \in \{3, 5, 7\}$.

---

## 3. Data Partitions & Separation of Tuning vs. Testing

To eliminate researcher degrees of freedom and p-hacking:
1. **Exploration & Hyperparameter Tuning Set**:
   - Random Seeds: `0` through `99`.
   - Used for tuning estimation window sizes, CUSUM detection thresholds, and cost weights.
2. **Held-Out Test Set**:
   - Random Seeds: `1000` through `1099` (50+ independent trajectories).
   - Evaluated exactly once under frozen parameters. No post-hoc tuning permitted on this partition.

---

## 4. Exclusion & Stopping Rules

- **Power Analysis & Sample Size**:
  - Minimum sample size: $N = 50$ independent trajectory seeds per scenario.
  - Required statistical power: $1 - \beta = 0.80$ at $\alpha = 0.05$ for a minimum detectable effect size of Cohen's $d = 0.35$.
- **Exclusion Rules**:
  - Trajectories where simulation fails due to memory exhaustion or non-convergence are flagged and logged, but never silently discarded.
  - No data point will be removed based on outcome.

---

## 5. Decision Gates

| Outcome on Held-Out Test Set | Scientific Action & Publication Framing |
| :--- | :--- |
| **Adaptive beats Rungs 3–4 (DEM / DGR)** ($p < 0.05$, CI excludes 0) | Core claim confirmed. Proceed to interleaved IBM Quantum hardware validation. |
| **Adaptive $\approx$ Rungs 3–4** ($p \ge 0.05$, small effect size) | Reframe as an empirical study on the boundaries of adaptation and estimation overhead. |
| **Adaptive loses to Static MWPM** ($\Delta < 0$) | Publish as an honest negative-result and benchmark study for open-source drift evaluation. |
