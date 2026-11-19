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

---

## 3. The Three-Gate Decision Algorithm

```text
================================================================================
Algorithm 1: Confidence-Gated Cost-Constrained Adaptive Decoder Selection (CG-CCADS)
================================================================================
Input: 
  Syndrome window s_t, Active decoder d_active, Baseline decoder d_base,
  Candidate decoder set D, Adaptation tax tau_adapt,
  Thresholds: T_drift, delta_disagree, epsilon_deadzone,
  Significance level alpha (z_{1-alpha}), Hysteresis margin T_hys,
  Probation window K_prob, Adaptation budget B_epoch.
Output:
  Selected control action a_t in {STAY, SWITCH(d*), ROLLBACK(d_base)}.

1:  // Step 1: Update Multi-Timescale Noise Estimators
2:  r_det <- MeanDefectRate(s_t)
3:  D_t, is_drift_flag <- UpdateCUSUM(r_det)
4:  
5:  // Step 2: Handle Active Probation Safety Window
6:  if InProbationState() then
7:      probation_counter <- probation_counter + 1
8:      if RecentSurrogatePerformance() < -Threshold_Failure then
9:          ResetCUSUM()
10:         Return ROLLBACK(d_base)   // Safety rollback to default baseline
11:     end if
12:     if probation_counter >= K_prob then
13:         CommitPermanentSwitch()
14:     end if
15: end if
16: 
17: // GATE 1: Drift Significance Gate
18: if D_t <= T_drift and not is_drift_flag then
19:     Return STAY   // Regarded as stationary Poisson noise; cost = 0
20: end if
21: 
22: // GATE 2: Relevance & Decoder Disagreement Gate
23: C_t <- ComputePairwiseDisagreement(s_t, d_active, d_candidate)
24: if C_t <= delta_disagree then
25:     Return STAY   // Candidates agree on > (1 - delta_disagree) corrections; zero benefit
26: end if
27: 
28: // GATE 3: Lower Confidence Bound & Cost Constraint Gate
29: delta_hat <- EstimateSurrogateGain(s_t, d_active, d_candidate)
30: sigma_delta <- ComputeStandardError(s_t)
31: LCB_gain <- delta_hat - z_{1-alpha} * sigma_delta
32: 
33: // Dead Zone & Hysteresis Checks
34: required_margin <- tau_adapt + epsilon_deadzone + (T_hys if d_active == d_base else 0)
35: 
36: if LCB_gain > required_margin and HasRemainingBudget(B_epoch) then
37:     ConsumeBudget(B_epoch)
38:     ResetCUSUM()
39:     InitializeProbation(K_prob)
40:     Return SWITCH(d_candidate)
41: else
42:     Return STAY   // Gain is statistically indistinguishable from cost; abstain
43: end if
================================================================================
```

---

## 4. Key Properties & Guarantees

1. **Zero Flapping Guarantee**: Under dual-threshold hysteresis $T_{\uparrow} > T_{\downarrow}$ and dead zone $\epsilon_{\text{deadzone}}$, high-frequency noise jitter cannot trigger rapid cyclical switching.
2. **Strict Budget Bound**: The controller never executes more than $B_{\text{max}}$ switches per epoch of length $L$, guaranteeing that switching latency cannot accumulate unboundedly.
3. **Probationary Safety**: Any switch enters a temporary probation period of $K$ blocks. If the candidate decoder performs worse than baseline under real hardware noise, the controller immediately executes `ROLLBACK`.
4. **Deployable Exclusivity**: All inputs $x_t$ are computed strictly from syndrome bitstrings and macro calibration data. Zero oracle information is ingested.
