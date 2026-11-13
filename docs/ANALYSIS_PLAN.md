# Pre-Registered Confirmatory Analysis Plan: Adaptive Quantum Error Correction Benchmark

**Document Type**: Pre-Registered Study Protocol & Confirmatory Analysis Plan (OSF / AsPredicted Format)  
**Status**: Formal Pre-Registration Specification  
**Version**: 2.0.0  
**Registration Date**: October 2026 (Internal Open-Science Specification; Frozen Prior to Confirmatory Suite Run)  
**Target Publication**: *IEEE Transactions on Quantum Engineering* / *Quantum Science and Technology*  
**Principal Investigator**: Pratham Singh  
**Cryptographic Provenance Lock**: `paper.lock` (SHA-256 verified against `configs/suite.yaml` and `data/manifest.json`)  
**Data Partitions**: Tuning & Calibration (Seeds 0–99); Confirmatory Held-Out (Seeds 1000–1029, $N=30$)

---

## 1. Study Overview & Administrative Information

### 1.1 Title & Scope
*"When Does Online Decoder Reweighting Pay Off in Quantum Error Correction? A Confidence-Gated Framework"*  
This document specifies the exact pre-registered analysis plan governing the confirmatory evaluation of the **Confidence-Gated Cost-Constrained Adaptive Decoder Selection (CG-CCADS)** framework. It defines the formal research hypotheses, sampling methodology, primary statistical endpoints, multiple-testing correction protocols, and explicit acceptance/rejection decision rules.

### 1.2 Open Science & Pre-Registration Standard
In accordance with open science guidelines (Nosek et al., 2018; AsPredicted.org):
1. **Temporal Precedence**: This analysis protocol, the underlying simulation circuits, and the supervisory controller hyperparameters were frozen in `configs/suite.yaml` prior to unblinding and analyzing the held-out confirmatory evaluation partition.
2. **Cryptographic Binding**: The configuration file `configs/suite.yaml` and dataset manifest `data/manifest.json` are cryptographically locked via SHA-256 digests in `paper.lock`.
3. **Zero Post-Hoc Tuning**: No hyperparameters (e.g., CUSUM threshold $T_{\text{drift}}$, significance level $\alpha$, deadzone $\epsilon_{\text{deadzone}}$, switching cost $C_{\text{switch}}$) may be modified after observing held-out partition outputs.

---

## 2. Research Hypotheses & Theoretical Foundations

### 2.1 Primary Confirmatory Hypothesis (H1: Localized Drift Adaptation Win)
Under localized, spatially asymmetric physical noise drift (e.g., a single drifting transmon qubit hotspot whose error rate jumps from baseline $0.4\%$ to $5.0\%$), an online adaptive decoder utilizing Spitz et al. (2018) covariance-parity re-estimation with confidence gating will achieve a statistically significant paired reduction in logical error rate (LER) relative to static factory-calibrated MWPM:
$$\text{LER}_{\text{CG-CCADS}} < \text{LER}_{\text{static-MWPM}} \quad \iff \quad \bar{\Delta} < 0, \quad p_{\text{Holm}} < 0.05$$
where $\bar{\Delta} \equiv \frac{1}{N} \sum_{i=1}^N (\text{LER}_{\text{CG-CCADS}, i} - \text{LER}_{\text{static-MWPM}, i})$ evaluated at primary window size $W=2{,}000$ and distance $d=3$.

### 2.2 Null Hypotheses & Negative Controls (H0: Safe Abstention)
1. **H0,1 (Stationary Baseline Negative Control)**: Under stationary physical Pauli noise ($p_e = 0.0040$ constant over time), decoder re-estimation has no genuine physical parameter drift to exploit. The optimal policy is safe abstention (`STAY`). The mean paired difference must be statistically indistinguishable from zero:
   $$\bar{\Delta}_{\text{stationary}} \approx 0, \quad p > 0.05, \quad \text{Stay Rate} \ge 90\%$$
2. **H0,2 (Uniform Global Drift Negative Control)**: When physical noise scales uniformly across all qubits and couplers ($p'_e = c \cdot p_e$), all DEM edge weights shift by an approximately constant additive factor $\Delta w_e \approx -\ln(c)$. Because additive shifts preserve Dijkstra shortest paths, online adaptation provides negligible headroom ($< 2\%$). The controller must safely abstain (`STAY`):
   $$\bar{\Delta}_{\text{uniform}} \approx 0, \quad p > 0.05, \quad \text{Stay Rate} \ge 85\%$$
3. **H0,3 (Small-Window Variance Trap Methodological Boundary)**: For small estimation windows ($W \le 1{,}000$, e.g., $W = 200$ shots), finite-sample coincidence noise causes unconstrained naive reweighting to severely degrade logical error rate ($+104.38\%$ inflation). CG-CCADS must safely abstain (`STAY`) in $\ge 90\%$ of decision epochs, protecting logical fidelity:
   $$\text{LER}_{\text{naive}} \gg \text{LER}_{\text{static}}, \quad \text{LER}_{\text{CG-CCADS}} \approx \text{LER}_{\text{static}}, \quad \text{Stay Rate}_{\text{CG-CCADS}} \ge 90\%$$

### 2.3 Secondary Confirmatory Hypothesis (H2: Below-Threshold Fault-Tolerance Scaling)
All evaluated code instances must operate below the fault-tolerant threshold ($p_{\text{base}} = 0.004 < p_{\text{th}} \approx 0.01$). Specifically, increasing rotated surface code distance from $d=3$ to $d=5$ must suppress logical error rate exponentially:
$$\Lambda_{3 \to 5} \equiv \frac{\text{LER}(d=3)}{\text{LER}(d=5)} > 1.0$$
confirming that adaptive reweighting preserves below-threshold error suppression.

---

## 3. Data Partitions & Protocol Partitioning

To strictly prevent p-hacking, overfitting, and post-hoc parameter adjustments, the random seed space is strictly partitioned into two disjoint sets:

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

## 4. Statistical Unit, Design, & Primary Endpoints

### 4.1 Statistical Unit of Analysis
The fundamental statistical unit is the **independent stochastic noise trajectory** ($N=30$ held-out seeds). 
- Each trajectory simulates $10{,}000$ consecutive syndrome extraction rounds on a rotated surface code patch under time-varying circuit-level Pauli noise.
- Every trajectory is evaluated across sliding observation windows $W \in \{200, 1{,}000, 2{,}000\}$ shots and distances $d \in \{3, 5\}$.

### 4.2 Paired Experimental Design
To eliminate inter-trajectory noise variance and maximize statistical power, an exact **paired-sample design** is enforced:
- For every seed $i \in \{1000, \dots, 1029\}$, the physical noise trajectory and syndrome bitstrings $X \in \{0, 1\}^{10000 \times |V|}$ are generated once.
- All competing decoder arms—(1) Static Factory MWPM, (2) Naive Reweighted MWPM, (3) CG-CCADS, and (4) Oracle Matched MWPM—decode the **exact same realization of physical circuit faults and syndrome bitstrings**.

### 4.3 Primary Endpoint: Trajectory-Paired Delta
For seed $i$, the primary endpoint is the paired logical error rate difference:
$$\Delta_i \equiv \text{LER}_{\text{CG-CCADS}, i} - \text{LER}_{\text{static-MWPM}, i}$$
where a negative delta ($\Delta_i < 0$) indicates superior logical memory retention under adaptive control.

---

## 5. Statistical Testing Framework & Multiple Testing Control

### 5.1 Primary Hypothesis Test: Two-Sided Paired Student's $t$-Test
The mean paired delta $\bar{\Delta} = \frac{1}{N} \sum_{i=1}^N \Delta_i$ and sample standard deviation $s_\Delta = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (\Delta_i - \bar{\Delta})^2}$ define the test statistic:
$$t = \frac{\bar{\Delta}}{s_\Delta / \sqrt{N}}$$
evaluated against Student's $t$-distribution with $\nu = N - 1 = 29$ degrees of freedom.

### 5.2 Confidence Intervals: Percentile Student's $t$
Two-sided $95\%$ Confidence Intervals for the mean paired delta are constructed via:
$$\text{CI}_{95\%} = \left[ \bar{\Delta} - t_{29, 0.975} \frac{s_\Delta}{\sqrt{N}}, \; \bar{\Delta} + t_{29, 0.975} \frac{s_\Delta}{\sqrt{N}} \right]$$
where $t_{29, 0.975} \approx 2.0452$.

### 5.3 Family-Wise Multiple-Testing Control: Holm-Bonferroni Step-Down
To guarantee that the global Family-Wise Error Rate across the family of confirmatory regimes is strictly bounded by $\alpha = 0.05$, $p$-values are adjusted using the **Holm-Bonferroni step-down procedure**:
1. Sort the $m$ family $p$-values in ascending order: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$.
2. At step $k$, compare $p_{(k)}$ to threshold $\alpha_k = \frac{\alpha}{m - k + 1}$.
3. If $p_{(k)} \le \alpha_k$, reject $H_{(k)}$ and continue; otherwise, retain all remaining hypotheses.

### 5.4 Non-Parametric Sensitivity Verification: Wilcoxon Signed-Rank Test
As a pre-registered sensitivity analysis against potential departures from normality in the delta distribution, the non-parametric **Wilcoxon signed-rank test** is conducted concurrently across all paired deltas. Rejection under both the parametric $t$-test and non-parametric Wilcoxon test is required for robust scientific confirmation.

---

## 6. Prospective Design Power Calculation (Sample Sizing)

*Note: This section records the prospective sample-sizing power calculation performed prior to data collection.*

### 6.1 Pilot Parameters on Tuning Partition (Seeds 0–99)
- Standard deviation of trajectory-paired deltas at $d=3, W=2{,}000$ under localized drift: $\sigma_\Delta \approx 0.0011$.
- Minimum clinically and practically relevant effect size: $\Delta_{\text{min}} = -0.0006$ (corresponding to a $\sim 2.5\%$ relative reduction in logical error rate).
- Standardized effect size (Cohen's $d$):
  $$d = \frac{|\Delta_{\text{min}}|}{\sigma_\Delta} = \frac{0.0006}{0.0011} \approx 0.545$$

### 6.2 Exact Power Derivation via Non-Central $t$-Distribution
Under sample size $N = 30$, degrees of freedom $\nu = 29$, and non-centrality parameter:
$$\delta_{\text{ncp}} = \sqrt{N} \cdot d = \sqrt{30} \times 0.545 \approx 5.477 \times 0.545 \approx 2.988$$
For two-sided critical value $t_{\text{crit}} = 2.0452$:
$$\text{Statistical Power} = 1 - \beta = 1 - F_{t, 29, 2.988}(2.0452) + F_{t, 29, 2.988}(-2.0452) \approx 0.841 \quad (84.1\%)$$

The sample size of $N = 30$ independent trajectories provides **$84.1\%$ statistical power** to detect the target effect size at significance level $\alpha = 0.05$.

---

## 7. Pre-Specified Decision Rules & Acceptance Matrix

The confirmatory acceptance of scientific claims is strictly determined by the outcome matrix below:

| Hypothesis / Benchmark Scenario | Observed Metric Requirement | Acceptance Threshold | Pre-Specified Scientific Decision |
| :--- | :--- | :---: | :--- |
| **Primary: Localized Hotspot (H1)** | Paired Delta $\bar{\Delta}$, Holm-adjusted $p$ | $\bar{\Delta} < 0$ and $p_{\text{Holm}} < 0.05$ | **Confirmed**: Online confidence-gated reweighting provides genuine physical QEC payoff. |
| **Negative Control: Stationary (H0,1)** | Paired Delta $\bar{\Delta}$, Stay Rate | $p > 0.05$ and Stay Rate $\ge 90\%$ | **Confirmed**: Controller provably abstains; immune to false-alarm noise corruption. |
| **Negative Control: Uniform Drift (H0,2)**| Paired Delta $\bar{\Delta}$, Stay Rate | $p > 0.05$ and Stay Rate $\ge 85\%$ | **Confirmed**: Controller recognizes uniform drift invariance and avoids futile switching. |
| **Method Boundary: Small Window (H0,3)**| Naive LER inflation, Stay Rate | $\text{LER}_{\text{naive}} > 1.5 \text{LER}_{\text{stat}}$, Stay $\ge 90\%$ | **Confirmed**: Finite-sample variance explosion demonstrated; abstention protects memory. |
| **Threshold Scaling: Distance (H2)** | Error suppression factor $\Lambda_{3 \to 5}$ | $\Lambda_{3 \to 5} > 1.0$ | **Confirmed**: All benchmark simulations operate safely below the fault-tolerant threshold. |

---

## 8. Data Management, Integrity Checkers, & Audit Protocol

1. **Automated AST Zero-Literals Verification**:
   The test script `tests/test_no_hardcoded_results.py` parses generator source files via Python's Abstract Syntax Tree (`ast`), verifying that zero empirical floats exist outside a structural coordinate whitelist.
2. **Cryptographic SHA-256 Lock**:
   Running `python scripts/verify_lock.py` verifies byte-for-byte SHA-256 hash identity across all 11 core data files, LaTeX tables, vector figures, and configuration files recorded in `paper.lock`.
3. **Disclosures of Negative and Retracted Data**:
   All historical claims, including the retraction of the synthetic binomial IBM Heron script (`examples/synthetic_ab_demo.py`) and preliminary unmitigated IBM Marrakesh characterization datasets, are recorded in `VALIDATION.md`.
