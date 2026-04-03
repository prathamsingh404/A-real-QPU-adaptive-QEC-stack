# Algorithm Specification: Confidence-Gated Cost-Constrained Adaptive Decoder Selection (CG-CCADS)

> **Document Type**: Manuscript Core Contribution (Algorithm 1)  
> **Status**: Verified Implementation in `src/adaptive_qec/controller/cg_ccads.py`  
> **Target Publication**: *IEEE Transactions on Quantum Engineering*, *PRX Quantum*  
> **Tasks Addressed**: TASK-039 (Point 25), Submission Gate B (Algorithmic Novelty)

---

## 1. Problem Formulation

Let $s_t \in \{0, 1\}^{M \times W}$ denote the syndrome defect matrix observed across $M$ detectors over observation window $t$ containing $W$ syndrome extraction cycles.

In an active quantum computer, the true logical error rate $P_L$ and true latent noise rates are fundamentally unobservable. The QEC controller must select a control action $a_t \in \mathcal{A}$ where:
$$\mathcal{A} = \{\text{STAY}\} \cup \{\text{SWITCH}(d) \mid d \in \mathcal{D}\}$$
subject to:
1. **Switching and Latency Cost**: Incurring adaptation tax $\tau_{\text{adapt}} = C_{\text{switch}} + C_{\text{latency}} + C_{\text{compute}} + C_{\text{physical}}$ whenever $a_t \neq \text{STAY}$.
2. **Epistemic Uncertainty**: Decisions must be justified under confidence bound $1 - \alpha$ using only deployable observables.

---

## 2. Mathematical Controller State

At each observation cycle $t$, the controller constructs the observable state vector:
$$x_t = \big[\hat{p}_X, \hat{p}_Z, \hat{p}_{2q}, \hat{r}_{\text{det}}, D_t, C_t, U_t, G_t\big]^T$$

Where:
* $\hat{p}_X, \hat{p}_Z$: Spatial defect fractions on $X$-type and $Z$-type stabilizers.
* $\hat{p}_{2q}$: Baseline two-qubit error rate from periodic calibration telemetry.
* $\hat{r}_{\text{det}}$: Mean detector firing rate $\hat{r}_{\text{det}} = \frac{1}{M \cdot W} \sum_{m, w} s_{t}[m, w]$.
* $D_t$: Multi-timescale drift score derived from fast EWMA and medium-term cumulative sum (CUSUM):
  $$z_t = \frac{\bar{r}_{\text{fast}, t} - \bar{r}_{\text{slow}, t}}{\hat{\sigma}_r}$$
  $$S_t = \max\big(0, S_{t-1} + |z_t| - k_{\text{drift}}\big)$$
  $$D_t = S_t$$
* $C_t$: Pairwise candidate decoder disagreement rate evaluated counterfactually on recent syndrome buffer:
  $$C_t(d_{\text{active}}, d_{\text{candidate}}) = \frac{1}{W} \sum_{w=1}^W \mathbb{I}\left[\hat{L}_{w, d_{\text{active}}} \neq \hat{L}_{w, d_{\text{candidate}}}\right]$$
* $U_t$: Empirical standard error of the performance estimator: $U_t = \hat{\sigma}_{\Delta} / \sqrt{W}$.
* $G_t$: Expected net gain $\hat{\Delta}_t = \mathbb{E}\left[\text{Reward}(d_{\text{candidate}}) - \text{Reward}(d_{\text{active}})\right]$.
