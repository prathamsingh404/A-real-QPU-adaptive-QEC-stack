# Confirmatory Pre-Registration Protocol: Adaptive Quantum Error Correction Benchmark

> **Document Status**: Active Pre-Registration Specification (Synchronized with [`docs/ANALYSIS_PLAN.md`](file:///docs/ANALYSIS_PLAN.md))  
> **Registration Standard**: Open Science Framework (OSF) / AsPredicted Formal Protocol  
> **Version**: 2.0.0 (Frozen Prior to Held-Out Confirmatory Unblinding)  
> **Principal Investigator**: Pratham Singh  
> **Cryptographic Provenance Lock**: `paper.lock` (SHA-256 verified against `configs/suite.yaml` and `data/manifest.json`)  
> **Partitioning**: Tuning / Calibration (Seeds 0–99); Confirmatory Held-Out (Seeds 1000–1029, $N=30$ Independent Trajectories)

---

## 1. Study Overview & Executive Summary

This pre-registration document establishes the formal confirmatory protocol for evaluating the **Confidence-Gated Cost-Constrained Adaptive Decoder Selection (CG-CCADS)** framework. It defines the formal research hypotheses, sampling methodology, primary statistical endpoints, multiple-testing correction protocols, and explicit acceptance/rejection decision rules.

The comprehensive technical specification, mathematical proofs, and literature derivations are maintained in [`docs/ANALYSIS_PLAN.md`](file:///docs/ANALYSIS_PLAN.md).

---

## 2. Formal Research Hypotheses

### 2.1 Primary Confirmatory Hypothesis (H1: Localized Drift Adaptation Win)
Under localized, spatially asymmetric physical noise drift (e.g., a drifting transmon qubit hotspot where error rate shifts from baseline $0.4\%$ to $5.0\%$), online covariance-parity re-estimation with confidence gating will achieve a statistically significant paired reduction in logical error rate (LER) relative to static factory-calibrated MWPM:
$$\text{LER}_{\text{CG-CCADS}} < \text{LER}_{\text{static-MWPM}} \quad \iff \quad \bar{\Delta} < 0, \quad p_{\text{Holm}} < 0.05$$
where $\bar{\Delta} \equiv \frac{1}{N} \sum_{i=1}^N (\text{LER}_{\text{CG-CCADS}, i} - \text{LER}_{\text{static-MWPM}, i})$ evaluated at primary window size $W=2{,}000$ and distance $d=3$.

### 2.2 Null Hypotheses & Negative Controls (H0: Safe Abstention)
1. **H0,1 (Stationary Baseline Negative Control)**: Under stationary physical Pauli noise ($p_e = 0.0040$ constant), decoder re-estimation has no genuine physical parameter drift to exploit. The optimal policy is safe abstention (`STAY`). The mean paired difference must be statistically indistinguishable from zero:
   $$\bar{\Delta}_{\text{stationary}} \approx 0, \quad p > 0.05, \quad \text{Stay Rate} \ge 90\%$$
2. **H0,2 (Uniform Global Drift Negative Control)**: When physical noise scales uniformly across all qubits and couplers ($p'_e = c \cdot p_e$), all DEM edge weights shift by an approximately constant additive factor $\Delta w_e \approx -\ln(c)$. Because additive shifts preserve Dijkstra shortest paths (Theorem 6.1), online adaptation provides negligible headroom ($< 2\%$). The controller must safely abstain (`STAY`):
   $$\bar{\Delta}_{\text{uniform}} \approx 0, \quad p > 0.05, \quad \text{Stay Rate} \ge 85\%$$
3. **H0,3 (Small-Window Variance Trap Methodological Boundary)**: For small estimation windows ($W \le 1{,}000$, e.g., $W = 200$ shots), finite-sample coincidence noise causes unconstrained naive reweighting to severely degrade logical error rate ($+104.38\%$ inflation). CG-CCADS must safely abstain (`STAY`) in $\ge 90\%$ of decision epochs, protecting logical fidelity:
   $$\text{LER}_{\text{naive}} \gg \text{LER}_{\text{static}}, \quad \text{LER}_{\text{CG-CCADS}} \approx \text{LER}_{\text{static}}, \quad \text{Stay Rate}_{\text{CG-CCADS}} \ge 90\%$$

### 2.3 Secondary Confirmatory Hypothesis (H2: Below-Threshold Fault-Tolerance Scaling)
All evaluated code instances must operate below the fault-tolerant threshold ($p_{\text{base}} = 0.004 < p_{\text{th}} \approx 0.01$). Specifically, increasing rotated surface code distance from $d=3$ to $d=5$ must suppress logical error rate exponentially:
$$\Lambda_{3 \to 5} \equiv \frac{\text{LER}(d=3)}{\text{LER}(d=5)} > 1.0$$
confirming that adaptive reweighting preserves below-threshold error suppression.

---

## 3. Strict Seed Partitioning & Experimental Hygiene

To strictly prevent p-hacking, data peeking, and post-hoc parameter adjustments, the random seed space is strictly partitioned into two disjoint sets:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              STRICT SEED PARTITIONING                                  │
│                                                                                        │
│  1. EXPLORATORY / TUNING PARTITION (Seeds 0 - 99):                                     │
│     ├── Purpose: Hyperparameter calibration, reference defect estimation,             │
│     │   pilot variance measurement, and code testing.                                  │
│     ├── Calibrated Parameters: T_drift = 3.0, alpha = 0.05, eps_deadzone = 0.005,      │
│     │   C_switch = 0.0005, tau_adapt = 0.0010.                                         │
│     └── Status: Completely isolated; EXCLUDED from all final manuscript tables.        │
│                                                                                        │
│  2. CONFIRMATORY HELD-OUT PARTITION (Seeds 1000 - 1029, N = 30 Trajectories):          │
│     ├── Purpose: Pre-registered hypothesis testing on unseen noise realizations.       │
│     ├── Execution: Executed exactly ONCE under frozen configs in configs/suite.yaml.   │
│     └── Status: Strictly locks all data in data/suite_results.json and paper.lock.     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Statistical Methodology & Endpoints

- **Statistical Unit**: Independent stochastic noise trajectory ($N=30$ held-out seeds). Each trajectory evaluates $10{,}000$ consecutive syndrome extraction rounds on rotated surface code patches under time-varying circuit-level Pauli noise.
- **Exact Paired Design**: For every seed $i \in \{1000, \dots, 1029\}$, all competing decoders (Static Factory MWPM, Naive Reweighted MWPM, CG-CCADS, and Oracle Matched MWPM) decode the identical physical circuit faults and syndrome bitstrings $X \in \{0, 1\}^{10000 \times |V|}$.
- **Primary Endpoint**: Trajectory-paired delta:
  $$\Delta_i \equiv \text{LER}_{\text{CG-CCADS}, i} - \text{LER}_{\text{static-MWPM}, i}$$
- **Hypothesis Testing**: Two-sided paired Student's $t$-test evaluated against $\nu = N - 1 = 29$ degrees of freedom.
- **Confidence Intervals**: Two-sided $95\%$ Percentile Student's $t$ confidence interval with critical multiplier $t_{29, 0.975} \approx 2.0452$.
- **Multiple-Testing Control**: Holm-Bonferroni step-down correction across confirmatory scenarios to guarantee global FWER $\le 0.05$.
- **Non-Parametric Sensitivity Verification**: Wilcoxon signed-rank test executed concurrently across all paired deltas.

---

## 5. Prospective Design Power Calculation

Prior to unblinding the held-out partition, sample size calculations were established using pilot variances on seeds 0–99:
- Pilot standard deviation of paired deltas at $d=3, W=2{,}000$: $\sigma_\Delta \approx 0.0011$.
- Minimum practically relevant effect size: $\Delta_{\text{min}} = -0.0006$ ($\sim 2.5\%$ relative LER reduction).
- Standardized effect size (Cohen's $d$):
  $$d = \frac{|\Delta_{\text{min}}|}{\sigma_\Delta} = \frac{0.0006}{0.0011} \approx 0.545$$
- Non-centrality parameter under $N=30$:
  $$\delta_{\text{ncp}} = \sqrt{N} \cdot d = \sqrt{30} \times 0.545 \approx 2.988$$
- Statistical power derivation via non-central $t$-distribution:
  $$\text{Power} = 1 - \beta = 1 - F_{t, 29, 2.988}(2.0452) + F_{t, 29, 2.988}(-2.0452) \approx 0.841 \quad (84.1\%)$$

Thus, $N=30$ independent trajectories guarantees **$84.1\%$ prospective statistical power** at significance level $\alpha = 0.05$.

---

## 6. Pre-Specified Decision Rules & Acceptance Matrix

| Hypothesis / Benchmark Scenario | Observed Metric Requirement | Acceptance Threshold | Pre-Specified Scientific Decision |
| :--- | :--- | :---: | :--- |
| **Primary: Localized Hotspot (H1)** | Paired Delta $\bar{\Delta}$, Holm-adjusted $p$ | $\bar{\Delta} < 0$ and $p_{\text{Holm}} < 0.05$ | **Confirmed**: Online confidence-gated reweighting provides genuine physical QEC payoff. |
| **Negative Control: Stationary (H0,1)** | Paired Delta $\bar{\Delta}$, Stay Rate | $p > 0.05$ and Stay Rate $\ge 90\%$ | **Confirmed**: Controller provably abstains; immune to false-alarm noise corruption. |
| **Negative Control: Uniform Drift (H0,2)**| Paired Delta $\bar{\Delta}$, Stay Rate | $p > 0.05$ and Stay Rate $\ge 85\%$ | **Confirmed**: Controller recognizes uniform drift invariance and avoids futile switching. |
| **Method Boundary: Small Window (H0,3)**| Naive LER inflation, Stay Rate | $\text{LER}_{\text{naive}} > 1.5 \text{LER}_{\text{stat}}$, Stay $\ge 90\%$ | **Confirmed**: Finite-sample variance explosion demonstrated; abstention protects memory. |
| **Threshold Scaling: Distance (H2)** | Error suppression factor $\Lambda_{3 \to 5}$ | $\Lambda_{3 \to 5} > 1.0$ | **Confirmed**: All benchmark simulations operate safely below the fault-tolerant threshold. |

---

## 7. Cryptographic Reproducibility Commands

To independently audit and verify this protocol against committed data artifacts:

```bash
# 1. Verify SHA-256 cryptographic provenance lock across all artifacts
python scripts/verify_lock.py

# 2. Run AST checker confirming zero hardcoded literals in paper tables
python -m pytest tests/test_no_hardcoded_results.py

# 3. Verify below-threshold distance scaling (H2) and latency budget
python -m pytest tests/test_empirical_scaling_and_ablation.py

# 4. Regenerate all formal LaTeX paper tables directly from data/suite_results.json
python src/adaptive_qec/analysis/make_paper_tables.py

# 5. Execute full test suite (346 tests)
python -m pytest -q
```\n