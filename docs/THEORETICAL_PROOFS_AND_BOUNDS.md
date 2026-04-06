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
