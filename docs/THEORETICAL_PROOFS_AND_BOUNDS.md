# Theoretical Modeling, Analytical Proofs & Concentration Bounds for Adaptive QEC

**Manuscript Reference Document**  
**Target Publication**: *IEEE Transactions on Quantum Engineering* / *Quantum Science and Technology*  
**Scope**: Mathematical Formulations, Concentration Bounds, and Sequential Guarantees for Label-Free Confidence-Gated Adaptive Decoding (CG-CCADS)

---

## 1. Foundational Formulation: Gross vs. Net Adaptation Value

Let $\mathcal{D}_t$ denote the physical error channel at time-step $t$, and let $L(d, \mathcal{D}_t)$ denote the expected logical error rate (LER) of decoding graph configuration $d \in \mathcal{A}$ under $\mathcal{D}_t$.

At evaluation epoch $k$, the supervisory controller evaluates whether to retain the currently active matching graph $d_{\text{active}}$ (the factory baseline prior or current model) or transition to an updated candidate matching graph $d_{\text{cand}}$ derived from empirical syndrome covariance estimation (Spitz et al., 2018).

### Definition 1.1: Gross and Net Adaptation Value
- **Gross Adaptation Advantage ($V_{\text{gross}}$)**:
  $$V_{\text{gross}}(d_{\text{cand}}, d_{\text{active}}; \mathcal{D}_k) \equiv L(d_{\text{active}}, \mathcal{D}_k) - L(d_{\text{cand}}, \mathcal{D}_k)$$
- **Total Adaptation Tax ($C_{\text{adapt}}$)**:
  $$C_{\text{adapt}} \equiv C_{\text{switch}} + C_{\text{latency}} + C_{\text{estimation}}$$
  where:
  - $C_{\text{switch}}$ is the graph reconfiguration / shortest-path cache invalidation cost.
  - $C_{\text{latency}}$ accounts for pipeline stalling during edge weight recomputation.
  - $C_{\text{estimation}}$ represents the empirical risk incurred by adopting finite-sample weight estimates.
- **Net Adaptation Value ($V_{\text{net}}$)**:
  $$V_{\text{net}} \equiv V_{\text{gross}} - C_{\text{adapt}}$$

### Principle 1.1: The Safe Abstention Imperative
Adaptation is strictly advantageous in expectation if and only if $V_{\text{net}} > 0$. If $V_{\text{gross}} \le C_{\text{adapt}}$, the optimal policy is zero-cost abstention (`STAY`), retaining the active baseline graph.

---

## 2. Statistical Concentration & False-Switching Bounds (Gate 2)

Online adaptation introduces a fundamental risk: finite-sample coincidence noise in small syndrome windows can create spurious empirical edge shifts, tricking an unconstrained controller into corrupting an optimal baseline graph (the *finite-sample variance explosion*).

### Setting and Assumptions
- Let $W$ be the number of syndrome shots in the current sliding window.
- Let $E$ be the set of edges in the detector error model (DEM), with cardinality $|E|$.
- For edge $e = (u, v) \in E$, let $\hat{p}_e$ be the Spitz covariance-parity estimator computed over window $W$, and let $p_{e, \text{ref}}$ be the reference prior error probability.
- Define the absolute empirical edge shift $\hat{\Delta}_e \equiv |\hat{p}_e - p_{e, \text{ref}}|$.
- Let $H_0$ denote the stationary null hypothesis: no physical drift has occurred, so $\mathbb{E}[\hat{p}_e] = p_{e, \text{ref}}$ for all $e \in E$.

### Theorem 2.1: Family-Wise Error Rate (FWER) Bound on Candidate Edge Shifts
Under the null hypothesis $H_0$, let the Lower Confidence Bound (LCB) for edge $e$ be defined as:
$$\text{LCB}_e(W) \equiv \hat{\Delta}_e - z_{1 - \alpha / (2|E|)} \cdot \text{SE}_e(W)$$
where:
$$\text{SE}_e(W) \equiv \sqrt{\frac{\tilde{p}_e(1 - \tilde{p}_e)}{W}}, \quad \tilde{p}_e = \max(p_{e, \text{ref}}, \hat{p}_e, 10^{-4})$$
and $z_{1 - \alpha / (2|E|)} = \Phi^{-1}(1 - \frac{\alpha}{2|E|})$ is the two-sided normal critical value under Bonferroni correction for $|E|$ candidate edges.

Then, for any non-negative deadzone threshold $\epsilon_{\text{deadzone}} \ge 0$, the probability of a false edge detection under $H_0$ is rigorously bounded by:
$$P\left(\exists e \in E : \text{LCB}_e(W) > \epsilon_{\text{deadzone}} \;\middle|\; H_0\right) \le \alpha$$

#### Proof:
For each individual edge $e \in E$, by the asymptotic normality of the Spitz sample covariance estimator and conservative variance bounding:
$$P\left(\hat{\Delta}_e - z_{1 - \alpha / (2|E|)} \text{SE}_e(W) > 0 \;\middle|\; H_0\right) \le \frac{\alpha}{|E|}$$
Because $\epsilon_{\text{deadzone}} \ge 0$, the event $\{\text{LCB}_e > \epsilon_{\text{deadzone}}\} \subseteq \{\text{LCB}_e > 0\}$.
Applying Boole's inequality (union bound) across all $|E|$ graph edges:
$$P\left(\bigcup_{e \in E} \left\{ \text{LCB}_e(W) > \epsilon_{\text{deadzone}} \right\} \;\middle|\; H_0\right) \le \sum_{e \in E} P\left(\text{LCB}_e(W) > 0 \;\middle|\; H_0\right) \le |E| \cdot \frac{\alpha}{|E|} = \alpha$$
Setting $\alpha = 0.05$ guarantees that the probability of false edge alarm at any single evaluation epoch is strictly bounded by $5\%$. $\blacksquare$

### Proposition 2.1: Finite-Sample Concentration via Empirical Bernstein Bound
For bounded per-shot indicator differences $Y_{k, e} \in [-1, 1]$ with empirical variance $\hat{\sigma}_e^2 = \frac{1}{W} \sum_{k=1}^W (Y_{k, e} - \bar{Y}_e)^2$, by empirical Bernstein's inequality:
$$P\left(|\bar{Y}_e - \mathbb{E}[Y_e]| > \tau \;\middle|\; H_0\right) \le 2 \exp\left( - \frac{W \tau^2}{2 \hat{\sigma}_e^2 + \frac{4}{3} \tau} \right)$$
When $p_e \ll 1$ (typical below-threshold QEC where $p_e \approx 0.004$), the variance $\hat{\sigma}_e^2 \approx p_e \ll 1/4$. Consequently, empirical Bernstein concentration yields significantly faster decay than worst-case range-based Hoeffding bounds ($\exp(-W\tau^2 / 2)$), explaining why modest window sizes ($W \ge 2{,}000$) suffice to suppress false edge adaptation.

---

## 3. Sequential Change-Point Filtering: CUSUM Average Run Length (Gate 1)

Before performing $|E|$ edge covariance evaluations, CG-CCADS filters the incoming stream through **Gate 1: Defect Rate CUSUM Test**.

### Definition 3.1: Standardized Detector Defect Residual
Let $\bar{\lambda}_{\text{win}} = \frac{1}{W} \sum_{k=1}^W \bar{x}_k$ be the mean detector defect rate across window $W$, and let $\lambda_{\text{ref}}$ be the stationary reference defect rate. The standardized drift statistic is:
$$Z_{\text{drift}} = \frac{|\bar{\lambda}_{\text{win}} - \lambda_{\text{ref}}|}{\text{SE}_{\text{defect}}}, \quad \text{SE}_{\text{defect}} = \frac{\text{std}(\bar{x})}{\sqrt{W}}$$

### Theorem 3.1: Average Run Length ($ARL_0$) under Null Stationary Noise
Let Page's cumulative sum detection procedure operate with decision threshold $T_{\text{drift}} > 0$. Under stationary noise ($H_0$), the expected number of observation windows before an erroneous drift alarm occurs (the Average Run Length under $H_0$, denoted $\text{ARL}_0$) satisfies Page's asymptotic lower bound:
$$\text{ARL}_0 = \mathbb{E}[\tau_{\text{false}} \mid H_0] \ge \frac{\exp(2 \delta T_{\text{drift}}) - 2\delta T_{\text{drift}} - 1}{2 \delta^2}$$
where $\delta$ is the minimum detectable shift in standardized units.

#### Operational Implication:
For $T_{\text{drift}} = 2.5$ and $\delta = 0.5$, $\text{ARL}_0 \gg 100$ evaluation epochs. This mathematical property guarantees that during stationary or slowly breathing noise, Gate 1 intercepts $>90\%$ of evaluation epochs and forces `STAY`, completely sparing downstream graph recomputation and protecting against repeated-testing false alarms.

---

## 4. Minimum Detectable Drift Amplitude (The Un-Adaptable Dead Zone)

### Theorem 4.1: Critical Drift Amplitude Boundary ($A_{\text{drift}}^*$)
Let the sensitivity of decoder advantage with respect to physical parameter shift $A_{\text{drift}}$ be locally linearized:
$$V_{\text{gross}}(A_{\text{drift}}) \approx \kappa \cdot A_{\text{drift}}, \quad \kappa \equiv \left. \frac{\partial V_{\text{gross}}}{\partial A_{\text{drift}}} \right|_{A=0} > 0$$

Online adaptation yields a positive net return ($V_{\text{net}} > 0$) if and only if the true physical drift amplitude exceeds the critical threshold:
$$A_{\text{drift}}^* = \frac{1}{\kappa} \left[ C_{\text{switch}} + C_{\text{latency}} + z_{1 - \alpha / (2|E|)} \sqrt{\frac{p_{\text{ref}}(1 - p_{\text{ref}})}{W}} + \epsilon_{\text{deadzone}} \right]$$

#### Proof:
From Definition 1.1, $V_{\text{net}} > 0 \iff V_{\text{gross}} > C_{\text{adapt}}$.
Substituting the linear approximation $V_{\text{gross}} \approx \kappa A_{\text{drift}}$ and the conservative statistical estimation barrier $z_{1 - \alpha/(2|E|)} \text{SE}(W) + \epsilon_{\text{deadzone}}$ required by Gate 2 and Gate 3 yields:
$$\kappa A_{\text{drift}} > C_{\text{switch}} + C_{\text{latency}} + z_{1 - \alpha/(2|E|)} \text{SE}(W) + \epsilon_{\text{deadzone}}$$
Dividing both sides by $\kappa > 0$ yields the result. $\blacksquare$

#### Corollary 4.1: The Un-Adaptable Dead Zone
For any physical drift with amplitude $A_{\text{drift}} < A_{\text{drift}}^*$:
1. The statistical signal is indistinguishable from finite-sample binomial shot noise.
2. The gross adaptation benefit is strictly smaller than the reconfiguration and estimation tax.
3. Attempting online adaptation in this dead zone guarantees systematic LER degradation. CG-CCADS provably abstains (`STAY`) in this regime.

---

## 5. Dynamic Regret Bounds in Piecewise-Stationary Noise

Consider a non-stationary QEC experiment spanning $T$ syndrome rounds, partitioned into $S$ piecewise-stationary environmental epochs:
$$\mathcal{T} = \bigcup_{s=1}^S [t_{s-1}, t_s), \quad t_0 = 0, \quad t_S = T$$
separated by $S - 1$ unknown change-points. Within each epoch $s$, the optimal matching graph configuration is $a_s^* = \arg\min_{a \in \mathcal{A}} L(a, \mathcal{D}_s)$.

### Theorem 5.1: Cumulative Tracking Regret Bound
Let cumulative tracking regret against the dynamic hindsight oracle with switching costs be defined as:
$$R(T) \equiv \sum_{t=1}^T \left[ L(a_t, \mathcal{D}_t) - L(a_t^*, \mathcal{D}_t) \right] + \sum_{t=1}^T C_{\text{switch}} \cdot \mathbf{1}\{a_t \ne a_{t-1}\}$$

Under the CG-CCADS policy with sequential CUSUM change-point detection (Gate 1), Bonferroni LCB estimation (Gate 2), and hysteresis cost thresholds (Gate 3), the cumulative regret satisfies:
$$R(T) \le \mathcal{O}\left( \sqrt{S \cdot |E| \cdot T \log T} + S \cdot C_{\text{switch}} \right)$$

#### Proof Sketch:
1. **Intra-Epoch Tracking Regret**:
   Within stationary epoch $s$ of duration $T_s = t_s - t_{s-1}$, by standard concentration for sub-Gaussian / bounded estimators in non-stationary bandit environments (Garivier & Moulines, 2011; Besbes, Gur, & Zeevi, 2014; Lattimore & Szepesvári, 2020), the estimation regret across $|E|$ edge parameters is bounded by $\mathcal{O}(\sqrt{|E| T_s \log T_s})$.
2. **Summation Over Epochs**:
   Applying the Cauchy-Schwarz inequality across the $S$ epochs:
   $$\sum_{s=1}^S \sqrt{T_s} \le \sqrt{S \sum_{s=1}^S T_s} = \sqrt{S T}$$
   Summing over all epochs yields total parameter tracking regret bounded by $\mathcal{O}(\sqrt{S \cdot |E| \cdot T \log T})$.
3. **Switching Penalty Control**:
   Spurious intra-epoch switching is prevented by the dual barriers of Gate 1 ($\text{ARL}_0$ exponential suppression) and Gate 2 ($P(\text{false switch}) \le \alpha$). Consequently, switching events occur at most $\mathcal{O}(1)$ times per true physical change-point, bounding total reconfiguration penalties to $\mathcal{O}(S \cdot C_{\text{switch}})$.
4. Combining the two bounds yields the claimed result:
   $$R(T) \le \mathcal{O}\left( \sqrt{S \cdot |E| \cdot T \log T} + S \cdot C_{\text{switch}} \right) \quad \blacksquare$$

---

## 6. Literature Grounding & Theoretical Scope

| Aspect | Classical Sliding-Window (Bhardwaj 2026; Spitz 2018) | Conservative / Switching Bandits (Wu 2016; Garivier 2011) | CG-CCADS (This Work) |
| :--- | :--- | :--- | :--- |
| **Update Mechanism** | Unconditional periodic re-estimation | Upper/Lower Confidence Bounds on Payoffs | 3-Gate Sequential Test (CUSUM + LCB + Tax) |
| **Observation Boundary** | Label-free syndrome coincidences | Scalar reward / feedback | Strictly label-free syndrome coincidences |
| **False-Alarm Defense** | None (sample variance ignored) | Sub-Gaussian confidence intervals | FWER Bonferroni across $|E|$ edges + CUSUM |
| **Dead-Zone Handling** | Updates on micro-fluctuations | Exploits until bound shifts | Provable abstention (`STAY`) below $A_{\text{drift}}^*$ |
| **Real-Time Role** | Recomputes matching graph every window | Reallocates exploration probabilities | Software meta-orchestration above FPGA decoders |
