# Comprehensive Scientific Audit & 129-Point Action Plan
## Safe & Selective Adaptive Quantum Error Correction (CG-CCADS)

> **Document Type**: Master Action Plan & Implementation Blueprint  
> **Status**: Active Scientific Specification  
> **Target Processor**: IBM Heavy-Hex Architecture (`ibm_marrakesh`, `ibm_kingston`, Heron r2)  
> **Simulation Infrastructure**: Stim, PyMatching v2, True Cluster-Growth Union-Find  
> **Primary Publication Target**: *IEEE Transactions on Quantum Engineering*, *PRX Quantum*, *Quantum Science and Technology*

---

## Executive Context: Transforming the Scientific Thesis

### The Motivating Negative Result (Never Erase)
In the 50,000-shot high-statistics benchmark (`experiments/results/adaptive_vs_static_high_stats_50k.json`):
- **Static MWPM**: $8,516 / 50,000 = 0.170320$ ($95\%$ Wilson CI: $[0.16705, 0.17364]$)
- **Naive Adaptive Policy**: $10,878 / 50,000 = 0.217560$ ($95\%$ Wilson CI: $[0.21396, 0.22120]$)
- **Net Performance**: The naive adaptive policy performed **$27.74\%$ worse** than static factory MWPM.

### The New Scientific Paradigm
Rather than making the generic (and occupied) claim:
> *"We built an adaptive QEC stack that dynamically switches decoders,"*

The new research question is:
> **"When is adaptation actually worth performing, and can a QEC controller decide—using only deployable, non-privileged observations—whether to adapt, which policy to select, or deliberately abstain (do nothing) under uncertainty and switching costs?"**

---

# Complete 129-Point Action Plan (Points 1 to 129)

### 1. Freeze the current project before changing anything
- **Mandate**: Create an immutable research snapshot of the v1 codebase before altering any algorithms or claims.
- **Repository Action**: Tag `v1-audit-historical` and maintain snapshot metadata in `research/v1_current_tip/`.
- **Target**: Ensure older benchmark JSONs are classified as `historical` or `retracted` rather than active claims.

### 2. Completely redesign the claims ledger
- **Mandate**: Replace informal markdown assertions with a machine-verifiable ledger schema (`VALIDATION.md` and `claims.json`).
- **Repository Action**: Each claim must link to a unique `claim_id`, `git_commit`, `evidence_artifacts`, and `pre_registered: bool`. Retracted claims must state the exact technical reason and superseding claim.
- **Verification**: Run `python scripts/verify_claims.py` in CI to fail on missing artifacts or stale claims.

### 3. Stop treating StandardExperimentResult as optional infrastructure
- **Mandate**: Enforce a unified, strongly-typed dataclass output across all experiment runners.
- **Repository Action**: Standardize `ExperimentManifest` (hashes, seeds, environment, git commit) and `ExperimentResult` (shots, LER, detection events, latency, switches).
- **Verification**: Prohibit loose dictionary dumping (`json.dump({"ler": ...})`) across all runner scripts.

### 4. Make experiment output immutable
- **Mandate**: Every experiment execution must write to a self-contained, hash-verified directory.
- **Repository Action**: Output directory structure: `results/EXP-{ID}/` containing `manifest.json`, `circuit.stim`, `detector_samples.bin`, `metrics.json`, `controller_trace.jsonl`, and `provenance.json` with SHA-256 digests.
- **Verification**: Any post-run modification to a result file invalidates its provenance hash.

### 5. Fix the hardware fallback immediately
- **Mandate**: Eliminate silent synthetic fallback when hardware execution fails in `src/adaptive_qec/runtime/qiskit_loop.py`.
- **Repository Action**: In `_run_circuit()`, replace `except Exception: return self._generate_synthetic_data(shots)` with raising a strict `HardwareExecutionError`.
- **Verification**: `execution_mode` must explicitly be `"hardware"`, `"simulation"`, or `"hardware_failed"`. Synthetic data is never labeled as hardware.

### 6. Build a strict controller observation boundary
- **Mandate**: Prevent any information leakage from the simulated noise generator into the controller.
- **Repository Action**: Enforce `ObservationFrame` containing only deployable observables: detector counts, spatial/temporal defect rates, syndrome spectrum, and calibration metadata.
- **Prohibited**: Controller must never receive `actual_p_x`, `true_leakage_state`, `burst_active`, or `future_noise`.

### 7. Test the boundary end-to-end
- **Mandate**: Construct automated tests that prove the controller is strictly invariant to hidden latent state changes.
- **Repository Action**: In `tests/test_architecture_boundaries.py`, test that identical detector streams with permuted latent noise labels produce bit-identical controller decisions.
- **Verification**: Static inspection and runtime interception ensure zero private field imports.

### 8. Replace detector-index assumptions with a real stabilizer map
- **Mandate**: Remove naive array bisections like `detectors[:, :n//2]` and `detectors[:, n//2:]`.
- **Repository Action**: Implement `CodeGeometry`, `DetectorMap`, and `StabilizerMap` mapping detector IDs to exact physical $(x, y, t)$ coordinates and stabilizer types ($X$ vs $Z$).
- **Verification**: Validated against surface code distance $d \in \{3, 5, 7\}$ geometries.

### 9. Separate decoder adaptation from physical intervention
- **Mandate**: Decouple classical decoder selection (MWPM, Union-Find, edge reweighting) from physical quantum circuit changes (Dynamical Decoupling, circuit depth, basis scheduling).
- **Repository Action**: Classical actions are evaluated counterfactually on the exact same syndrome data; physical actions require randomized, interleaved physical executions.

### 10. Create a real baseline ladder
- **Mandate**: Benchmark against a rigorous hierarchy of standard baselines, not just a weak unoptimized MWPM.
- **Ladder**:
  - `B0`: Static Factory MWPM (nominal factory calibration weights)
  - `B1`: Optimized Static MWPM (best fixed weights fit to baseline dataset)
  - `B2`: Genuine Cluster-Growth Union-Find
  - `B3`: Online Reweighted MWPM (syndrome-based sliding window, e.g., DGR / Bhardwaj style)
  - `B4`: Naive Adaptive Policy (current reactive controller, motivating baseline)
  - `B5`: Selective Adaptive Controller (CG-CCADS)
  - `B6`: Oracle Policy (sees true latent noise, defines theoretical upper bound)

### 11. Make the Union-Find implementation genuinely Union-Find
- **Mandate**: Ensure the Union-Find decoder is true cluster growth (grow $\to$ merge $\to$ neutralize $\to$ boundary peeling), not greedy heuristic matching.
- **Repository Action**: Audit `src/adaptive_qec/decoders/union_find.py` against Delfosse & Nickerson (2021). Add validation tests on high-weight cluster syndromes.

### 12. Build the new central algorithm (CG-CCADS)
- **Mandate**: Implement Confidence-Gated Cost-Constrained Adaptive Decoder Selection.
- **Repository Action**: Adapt only when: (1) Drift is statistically significant, (2) Candidate decoders exhibit disagreement, (3) Expected performance gain lower bound exceeds switching and latency costs.

### 13. Define the controller state
- **Mandate**: Construct explicit mathematical state vector $x_t = [\hat{p}_X, \hat{p}_Z, \hat{p}_{2q}, \hat{r}_{\text{det}}, D_t, C_t, U_t, G_t]$.
- **Components**: Estimated Pauli rates, detection density, drift magnitude $D_t$, decoder disagreement $C_t$, uncertainty $U_t$, and net value gain $G_t$.

### 14. Build a multi-timescale estimator
- **Mandate**: Combine fast change-point detection with medium-term sliding window estimation and slow baseline tracking.
- **Architecture**:
  - Short timescale: EWMA for fast anomaly alerts.
  - Medium timescale: CUSUM / SPRT for change-point confirmation.
  - Long timescale: Sliding-window covariance estimation for stable noise parameter estimation.

### 15. Add decoder disagreement as a decision signal
- **Mandate**: Run candidate decoders (MWPM, RW-MWPM, UF) on identical syndrome blocks to calculate disagreement rate $C_t = \frac{\# \text{pairwise discordant corrections}}{\# \text{pairs}}$.
- **Scientific Logic**: High drift with zero decoder disagreement implies changing decoders provides negligible benefit; adaptation is only worthwhile when candidates produce divergent corrections.

### 16. Add the explicit STAY (Abstain) action
- **Mandate**: Elevate `STAY` (deliberate abstention) to a first-class, selectable controller action.
- **Properties**: Switching cost $C_{\text{switch}} = 0$, latency penalty $C_{\text{latency}} = 0$. In stable or unresolvable noise regimes, `STAY` must be the optimal choice.

### 17. Replace the reward $1 - \text{LER}$ with deployable signals
- **Mandate**: Remove true logical error rate from online reward computation, as LER is unavailable during arbitrary quantum computation.
- **Repository Action**: Use observable surrogate signals: detection event entropy, syndrome autocorrelation, decoder disagreement, and offline calibration estimates. Keep true LER strictly for evaluation mode.

### 18. Create a deployable surrogate objective
- **Mandate**: Define an empirical risk objective $R_t = -(\alpha D_t + \beta C_t + \gamma U_t)$ coupled with offline policy value models $\hat{\Delta}_{a, b} = \hat{L}_b - \hat{L}_a$ parameterized by calibration telemetry.

### 19. Use a confidence bound, not a point estimate
- **Mandate**: Base switching decisions on conservative lower confidence bounds.
- **Rule**: Switch to candidate policy $a_{\text{new}}$ only if:
  $$\text{LCB}(\Delta) = \hat{\Delta} - z_{1-\alpha} \sigma_{\Delta} > C_{\text{switch}} + C_{\text{latency}}$$
  Otherwise, select `STAY`.

### 20. Do not collapse everything into arbitrary weights
- **Mandate**: Replace arbitrary linear penalty sums ($\lambda_1 C_{\text{switch}} + \lambda_2 C_{\text{latency}}$) with constrained optimization:
  $$\min_a \hat{L}(a) \quad \text{s.t.} \quad C_{\text{switch}} \le C_{\max}, \; C_{\text{latency}} \le L_{\max}, \; P(\text{harmful switch}) < \alpha$$

### 21. Add hysteresis to prevent flapping
- **Mandate**: Define dual switching thresholds ($T_{\uparrow} > T_{\downarrow}$) to eliminate high-frequency policy oscillation between decoders.
- **Metrics**: Track switch density, average dwell time, and false-switching frequency.

### 22. Add a rollback mechanism
- **Mandate**: If a newly activated decoder's observed performance deteriorates during operation, automatically trigger a safe rollback to the previous stable baseline.
- **State Machine**: `STABLE` $\to$ `EVALUATING` $\to$ `SWITCHING` $\to$ `VALIDATING` $\to$ `ACTIVE` or `ROLLBACK`.

### 23. Create a probationary switching period
- **Mandate**: When a new decoder policy is engaged, place it on probation for $K$ rounds. During probation, failure to meet expected risk bounds triggers rollback before policy becomes permanent.

### 24. Add a dead zone
- **Mandate**: Enforce a minimal improvement threshold $\epsilon$. If $|\hat{\Delta}| < \epsilon$, the controller forces `STAY`, refusing to switch for marginal, statistically indistinguishable gains.

### 25. Create the full controller pseudocode
- **Mandate**: Formalize the complete CG-CCADS decision flow in published, reviewable pseudocode integrating drift detection, disagreement, uncertainty estimation, confidence gating, budget validation, and rollback.

### 26. The really interesting research experiment (Regime Mapping)
- **Mandate**: Shift focus from a single scalar LER comparison to a multidimensional phase diagram mapping drift frequency vs drift amplitude vs switching cost.

### 27. Add adaptation value metrics
- **Mandate**: Formally calculate gross adaptation value $V_{\text{adapt}} = L_{\text{current}} - L_{\text{adapted}}$ and net adaptation value $V_{\text{net}} = V_{\text{adapt}} - C_{\text{switch}}$, evaluating where $V_{\text{net}} > 0$.

### 28. Build counterfactual decoding properly
- **Mandate**: Feed the exact same physical or simulated syndrome bitstream into parallel decoder instances to eliminate environmental variance in decoder comparisons.

### 29. Separate two counterfactual experiments
- **Mandate**: Distinguish:
  - (a) *Decoder-only counterfactual*: Same syndrome stream, different classical decoding algorithms (causally identical quantum state).
  - (b) *Hardware-intervention counterfactual*: Dynamical decoupling and circuit changes require interleaved, randomized experimental blocks ($A/B/A/B$).

### 30. Completely rebuild the leakage model
- **Mandate**: Deprecate naive detector-bit flipping (`detector[2] = 1`).
- **Repository Action**: Implement a stateful, physical phenomenological leakage model where leakage persists across rounds, propagates through two-qubit gates, and heralds leakage via measurement/reset.
