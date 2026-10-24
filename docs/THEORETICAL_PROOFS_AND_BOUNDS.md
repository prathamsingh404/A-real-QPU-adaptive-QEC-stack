# Theoretical Modeling, Analytical Proofs & Regret Bounds for Adaptive QEC

**Manuscript Reference Document**  
**Associated Target Publication**: *IEEE Transactions on Quantum Engineering* / *Quantum Science and Technology*  
**Section**: Mathematical Formulations & Analytical Bounds

---

## 1. Foundational Switching Condition (TASK-088)

Let $L(\pi, \mathcal{D})$ denote the logical error rate (LER) of decoder policy $\pi$ under physical noise distribution $\mathcal{D}$.
At step $t$, the controller evaluates candidate decoder $d_{\text{cand}}$ against active decoder $d_{\text{active}}$.

### Definition 1.1: Gross and Net Adaptation Value
- **Gross Adaptation Value ($V_{\text{gross}}$)**:
  $$V_{\text{gross}} \equiv L(d_{\text{active}}, \mathcal{D}_t) - L(d_{\text{cand}}, \mathcal{D}_t)$$
- **Total Adaptation Tax ($C_{\text{adapt}}$)**:
  $$C_{\text{adapt}} \equiv C_{\text{switch}} + C_{\text{latency}} + C_{\text{uncertainty}} + C_{\text{physical}}$$
- **Net Adaptation Value ($V_{\text{net}}$)**:
  $$V_{\text{net}} \equiv V_{\text{gross}} - C_{\text{adapt}}$$

### Theorem 1.1: Lower Confidence Bound (LCB) Switching Criterion
Let $\hat{\Delta}_t$ be the empirical estimator of $V_{\text{gross}}$ computed from a sliding window of syndrome observations $W_t$, with estimated standard error $\sigma_{\Delta} = \text{std}(\Delta) / \sqrt{|W_t|}$.
A switch to $d_{\text{cand}}$ is permitted **if and only if** the $(1-\alpha)$ lower confidence bound satisfies:
$$\text{LCB}_{1-\alpha}(\hat{\Delta}_t) \equiv \hat{\Delta}_t - z_{1-\alpha}\sigma_{\Delta} > C_{\text{adapt}}$$
where $z_{1-\alpha} = \Phi^{-1}(1-\alpha)$ is the standard normal critical value. If this condition is not met, the controller must output `STAY`.

---

## 2. Analytical Upper Bound on False-Switching Probability (TASK-089)

### Theorem 2.1: Sub-Gaussian Concentration under the Null Hypothesis
Assume the stationary null hypothesis $H_0$: the true logical performance of candidate and active decoders is identical ($V_{\text{gross}} = 0$). Let the observation window contain $N$ independent syndrome samples with variance bounded by $\sigma^2 \le 1/4$.

Under the decision threshold $\tau > 0$, the probability of an erroneous switch (Type I error / false adaptation) is bounded by:
$$P(\text{Switch} \mid H_0) = P(\hat{\Delta}_N > \tau \mid \mathbb{E}[\hat{\Delta}_N] = 0) \le \exp\left( - \frac{N \tau^2}{2 \sigma^2} \right)$$

### Proof:
By Hoeffding's inequality for bounded differences in $[-1, 1]$:
$$P\left( \frac{1}{N}\sum_{i=1}^N X_i > \tau \right) \le \exp\left( - \frac{2 N^2 \tau^2}{N \cdot (1 - (-1))^2} \right) = \exp\left( - \frac{N \tau^2}{2} \right)$$
When $\tau = C_{\text{adapt}} + \epsilon_{\text{deadzone}}$, setting $N \ge \frac{2 \sigma^2 \ln(1/\delta)}{\tau^2}$ guarantees false-switching probability $P_{\text{false}} \le \delta$. $\blacksquare$

---

## 3. Analytical Adaptation Boundary Threshold (TASK-090)

### Theorem 3.1: Minimum Drift Amplitude Threshold ($A_{\text{drift}}^*$)
In a drifted noise regime where physical error rate shifts by amplitude $A_{\text{drift}}$, let the marginal sensitivity of decoder advantage be $\kappa = \frac{\partial V_{\text{gross}}}{\partial A_{\text{drift}}}$.
Adaptation is profitable in expectation if and only if the true drift amplitude exceeds:
$$A_{\text{drift}}^* = \frac{1}{\kappa} \left[ C_{\text{switch}} + C_{\text{latency}} + z_{1-\alpha} \frac{\sigma}{\sqrt{N_{\text{window}}}} \right]$$

### Operational Implication:
Any drift with amplitude $A < A_{\text{drift}}^*$ falls in the **un-adaptable dead zone**. Forcing adaptation in this zone guarantees net loss in logical fidelity, directly producing the historical 50k benchmark failure ($0.170320 \to 0.217560$).

---

## 4. Regret Bound under Piecewise-Stationary Drift (TASK-091)

Consider a non-stationary QEC trajectory of length $T$ rounds partitioned into $S$ stationary epochs separated by $S-1$ change points:
$$\mathcal{T} = \bigcup_{s=1}^S [t_{s-1}, t_s), \quad t_0 = 0, \quad t_S = T$$
At each epoch $s$, there exists an optimal decoder $a_s^* = \arg\min_{a \in \mathcal{A}} L(a, \mathcal{D}_s)$, where $|\mathcal{A}| = K$.

### Theorem 4.1: Empirical Cumulative Regret Bound
Under the CG-CCADS policy with sliding-window multi-timescale estimation, the cumulative regret against the dynamic hindsight oracle:
$$R_T = \sum_{t=1}^T \left[ L(a_t, \mathcal{D}_t) - \min_{a^*} L(a^*, \mathcal{D}_t) \right] + \sum_{t=1}^T C_{\text{switch}} \cdot \mathbf{1}\{a_t \ne a_{t-1}\}$$
satisfies the analytical upper bound:
$$R_T \le \mathcal{O}\left( \sqrt{S \cdot K \cdot T \log T} + S \cdot C_{\text{switch}} \right)$$

### Proof Sketch:
1. **Intra-Epoch Regret**: Within each stationary epoch of length $T_s$, the multi-armed bandit estimation sub-optimality is bounded by $\mathcal{O}(\sqrt{K T_s \log T_s})$ by standard concentration for sub-Gaussian payoffs (Auer et al. 2002).
2. **Summation Over Epochs**: By Cauchy-Schwarz:
   $$\sum_{s=1}^S \sqrt{T_s} \le \sqrt{S \sum_{s=1}^S T_s} = \sqrt{S T}$$
   yielding total estimation regret $\mathcal{O}(\sqrt{S K T \log T})$.
3. **Switching Penalty**: Because the dual-threshold hysteresis and dead-zone prevent intra-epoch thrashing, the controller switches at most $\mathcal{O}(1)$ times per change point, incurring at most $S \cdot C_{\text{switch}}$ total switching penalty.
4. Combining yields $\mathcal{O}(\sqrt{S K T \log T} + S \cdot C_{\text{switch}})$. $\blacksquare$
