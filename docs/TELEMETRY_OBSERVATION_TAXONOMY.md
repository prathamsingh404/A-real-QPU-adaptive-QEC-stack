# Telemetry, Observation, and Action Taxonomy for Adaptive QEC

> **Status**: Manuscript Specification & Architectural Boundary Definition  
> **Target Audience**: Reviewers, Experimentalists, and Software Implementers  
> **Relevant Submission Gates**: Gate A (Scientific Correctness)  
> **Tasks Addressed**: TASK-016 (Point 9), TASK-017 (Point 37)

---

## 1. Executive Summary & Boundary Invariant

A fatal flaw in early adaptive QEC literature is **information leakage**: controllers conditioning decisions on unobservable ground truth (such as true latent Pauli error rates $p_X, p_Z$, exact cosmic ray injection schedules, or true logical error rates $P_L$).

In a physical fault-tolerant quantum computer executing an unknown user algorithm, the true logical error rate and instantaneous microscopic noise states are **fundamentally unobservable**. Only classical detector syndromes, macroscopic calibration snapshots, and classical decoder execution statistics can be observed online.

Furthermore, **classical decoder adaptation** (which operates post-measurement on classical bitstreams) must be strictly distinguished from **physical quantum circuit interventions** (which alter pulses, timing, or basis scheduling on the physical qubits).

---

## 2. Telemetry Observability Taxonomy Table

The following table explicitly demarcates the four tiers of data within our architecture:

| Tier | Category | Quantities / Signals | Source | Online Controller Allowed? | Purpose / Boundary Constraint |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Tier 0** | **Latent / Ground Truth (Oracle)** | - True instantaneous Pauli error rates ($p_X, p_Z, p_Y$)<br>- Simulated cosmic ray burst times & radii<br>- Exact qubit leakage flags ($|2\rangle$ state occupation)<br>- True Logical Error Rate ($P_L$)<br>- Noise schedule generator parameters | Simulation environment generator (`NoiseSchedule`) | ❌ **PROHIBITED** | Used strictly offline in `B6: Oracle` baseline for theoretical upper-bound comparison. **Never exposed to online controller.** |
| **Tier 1** | **Physical QPU Hardware Telemetry** | - IBM backend calibration snapshots ($T_1, T_2$ relaxation times)<br>- Readout error probabilities ($p_{\text{RO}}$)<br>- Two-qubit gate error rates ($p_{\text{CX}}, p_{\text{CZ}}$)<br>- Physical temperature & cryostat logs<br>- Backend calibration timestamp | QPU calibration service / Qiskit Backend | ⚠️ **READ-ONLY / SLOW** | Available only at coarse macroscopic intervals (hours/days). Read-only; cannot track fast microsecond drift. |
| **Tier 2** | **Deployable Syndrome Telemetry (`ObservationFrame`)** | - Detector firing counts & defect rate ($\hat{r}_{\text{det}} = D / (M \cdot S)$)<br>- Spatial defect fractions: $X$-type ($\hat{r}_X$) vs $Z$-type ($\hat{r}_Z$)<br>- Spatial defect center-of-mass & cluster radius<br>- Short-timescale EWMA detector rate & z-score<br>- Medium-timescale CUSUM / SPRT change-point statistics<br>- Pairwise candidate decoder disagreement rate ($C_t$) | Real-time classical syndrome extraction buffer | ✅ **PRIMARY CONTROLLER INPUT** | **Fully deployable in real-world FTQC.** Computed strictly from measurement bitstrings $\{0, 1\}^{M \times S}$. |
| **Tier 3** | **Decoder Execution Telemetry** | - Decoder execution latency ($T_{\text{decode}}$ in $\mu\text{s}$)<br>- Graph size / number of defect nodes $\|V\|$<br>- Peel / blossom step counts | Classical decoding engine | ✅ **CONTROLLER INPUT (COST / BUDGET)** | Feeds into the adaptation tax constraint ($\tau_{\text{adapt}} = C_{\text{switch}} + C_{\text{latency}}$). |

---

## 3. Intervention Separation: Classical vs Physical Actions

To avoid confounding experimental results, all control actions are formally classified into two mutually exclusive regimes:

### 3.1. Classical Decoder-Only Interventions (`CLASSICAL_ONLY`)
* **Definition**: Interventions that operate entirely in classical post-processing after syndrome measurements are extracted.
* **Examples**:
  - Switching between Factory MWPM ($B_0$) and Genuine Union-Find ($B_2$).
  - Updating edge weights in Online Reweighted MWPM ($B_3$).
  - Selecting Confidence-Gated Selective Selection (CG-CCADS, $B_5$).
  - Deliberately choosing `STAY` (abstaining from switching, $C_{\text{switch}} = 0$).
* **Evaluation Protocol**:
  - **Counterfactual Replay**: Classical policies MUST be evaluated on the **exact same syndrome bitstream** ($s_{1:N}$). This yields zero quantum variance and enables trajectory-paired statistical power ($\Delta_i = \text{LER}_{\text{adaptive}, i} - \text{LER}_{\text{static}, i}$).
