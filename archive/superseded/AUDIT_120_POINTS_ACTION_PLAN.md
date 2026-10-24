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

### 31. Add leakage propagation
- **Mandate**: Model leakage transfer during two-qubit operations: when a leaked qubit interacts via CZ/CNOT, it induces correlated Pauli faults on its partner and excites neighboring detector clusters.

### 32. Do not make leakage and DD the central novelty simultaneously
- **Mandate**: Keep the core paper claim focused on selective, cost-constrained adaptation under uncertainty. Treat leakage as an adversarial stress-test regime and DD as a physical intervention extension.

### 33. Rebuild the Dynamical Decoupling (DD) experiment
- **Mandate**: Test DD sequences (Free Evolution, XY4, XY8, CPMG) under rigorous randomized block assignments ($A/B/C/A/C/B$) on the same physical qubits, tracking pulse count and gate duration overhead.

### 34. Do not interpret higher detection rate automatically as better
- **Mandate**: Decouple detection event rate $R_{\text{det}}$ from logical error rate LER and state fidelity. Clearly explain that increased detector activity can signal pulse damage rather than enhanced sensitivity.

### 35. Hardware experiment design
- **Mandate**: Hold constant backend, qubit subset, transpilation optimization, and measurement maps. Interleave control policies across time to avoid confounding temporal drift with algorithmic performance.

### 36. Make calibration snapshots part of the analysis
- **Mandate**: Ingest and log full IBM backend calibration snapshots ($T_1, T_2$, readout error, gate errors) before and after each experimental block.

### 37. Separate "telemetry available" from "telemetry used"
- **Mandate**: Create an explicit observability table in the manuscript detailing what telemetry is physically readable vs what is provided to the controller.

### 38. Benchmark actual controller latency correctly
- **Mandate**: Profile every segment of the control loop: syndrome extraction, decoder execution, controller decision, and communication. Report $p50, p90, p95, p99$ and throughput (shots/second).

### 39. Build a real-time architecture separately
- **Mandate**: Honestly distinguish macro-timescale inter-batch orchestration (Python/cloud, seconds latency) from hypothetical microsecond intra-circuit FPGA feedforward.

### 40. Introduce a prediction layer only after the core controller works
- **Mandate**: Verify reactive selective control before implementing predictive forecasting models. Do not jump to complex neural forecasting prematurely.

### 41. Design the predictive controller
- **Mandate**: Evaluate whether anticipating structured drift (e.g., slow sinusoidal drift) via EWMA/AR(1) reduces adaptation lag without introducing speculative instability.

### 42. Use oracle experiments properly
- **Mandate**: Implement an Oracle controller (with privileged access to latent noise states) strictly as an analytical benchmark to quantify the maximum achievable theoretical adaptation gain.

### 43. Turn partial observability into a measured quantity
- **Mandate**: Define adaptation efficiency $\eta_{\text{adapt}} = G_{\text{observable}} / G_{\text{oracle}}$, measuring what fraction of oracle advantage is recovered using only deployable observables.

### 44. Build an adaptation regret metric
- **Mandate**: Track empirical cumulative regret $R_T = \sum_{t=1}^T L_t(a_t) - \sum_{t=1}^T L_t(a_t^*)$ against the hindsight optimal policy across non-stationary trajectories.

### 45. Fix the statistical unit completely
- **Mandate**: Cease treating individual shots within the same drift trajectory as i.i.d. The primary independent statistical unit must be the independent noise trajectory.

### 46. Use at least 20 independent trajectories for the core study
- **Mandate**: Base all headline simulation claims on $\ge 20$ (ideally 30) independent noise trajectories per scenario, reporting trajectory-level confidence intervals.

### 47. Use paired trajectories
- **Mandate**: For every seed, execute candidate controllers on the exact same noise trajectory instance, computing paired differences $\Delta_i = \text{LER}_{\text{adaptive}, i} - \text{LER}_{\text{static}, i}$ to eliminate environmental variance.

### 48. Split tuning and evaluation seeds
- **Mandate**: Enforce strict seed partitioning: Train (seeds 0–9), Validation (seeds 10–19), Test (seeds 20–49). Hyperparameters are frozen before running on held-out test seeds.

### 49. Predefine the complete noise suite
- **Mandate**: Pre-register 12 benchmark noise regimes: Stationary, Slow Drift, Fast Drift, Random Walk, Step Change, Periodic Bursts, Rare Catastrophic Bursts, Spatial Anisotropy, Temporal Correlation, Leakage Proxy, Physical Stateful Leakage, and Mixed Drift+Leakage.

### 50. Keep the system below threshold for the main performance study
- **Mandate**: Ensure physical error rates in the benchmark suite remain below the fault-tolerant threshold ($\sim 0.7\% - 1.0\%$) so error suppression with code distance $d$ is physically meaningful.

### 51. Add distance scaling sweeps ($d = 3, 5, 7$)
- **Mandate**: Evaluate all baselines across surface code distances $d \in \{3, 5, 7\}$, reporting error suppression factor $\Lambda = \text{LER}(d) / \text{LER}(d+2)$.

### 52. Measure more than LER
- **Mandate**: Report comprehensive metric tables: LER, logical fidelity, switch count, false-switch rate, rollback frequency, adaptation delay, noise estimation RMSE, decoder latency, and gate/pulse overhead.

### 53. Make the Pareto frontier a major paper figure
- **Mandate**: Plot 2D trade-off curves: LER vs Classical Latency and LER vs Adaptation Switching Overhead, showing where selective adaptation dominates naive strategies.

### 54. Build an adaptation phase diagram
- **Mandate**: Construct 2D phase diagrams over Drift Rate vs Drift Amplitude, coloring regions where `STAY`, `REWEIGHT`, `SWITCH`, and `ROLLBACK` are optimal.

### 55. Test the controller's decision accuracy
- **Mandate**: Evaluate the controller as a binary classifier of "adaptation worthiness," reporting precision, recall, false-adaptation rate, and missed-benefit rate.

### 56. Add calibration plots
- **Mandate**: Validate that predicted improvement probabilities align with empirical benefit frequencies (reliability diagram).

### 57. Consider conformal prediction for uncertainty bounds
- **Mandate**: Implement distribution-free conformal prediction intervals for estimated policy performance as an advanced confidence-gating mechanism.

### 58. Add an explicit switching budget
- **Mandate**: Introduce `AdaptationBudget(max_switches_per_trajectory, max_latency_penalty)`, demonstrating that selective adaptation saturates performance under tight budgets.

### 59. Add an "adaptation tax" model
- **Mandate**: Quantify the total cost of adaptation $\tau_{\text{adapt}} = C_{\text{switch}} + C_{\text{latency}} + C_{\text{compute}} + C_{\text{physical}}$ and prove adaptation is only rational when $V_{\text{adapt}} > \tau_{\text{adapt}}$.

### 60. Keep bandits, but move them inside the candidate-selection layer
- **Mandate**: Refactor Exp3/DASE bandit logic to propose candidate policies within the uncertainty-gated decision pipeline, subject to switching cost constraints.

### 61. Add contextual features to the policy selector
- **Mandate**: Incorporate contextual telemetry: drift velocity, detection event entropy, spatial asymmetry, syndrome autocorrelation, and time since last switch.

### 62. Compare three adaptation strategies directly
- **Mandate**: Benchmark: (1) Always-Adapt (Naive), (2) Threshold-Adapt (Simple CUSUM heuristic), and (3) Selective Confidence/Cost-Gated Adapt (CG-CCADS).

### 63. Optional predictive selective adaptation
- **Mandate**: Benchmark predictive CG-CCADS against reactive CG-CCADS specifically under structured sinusoidal drift.

### 64. Keep RL out of the core paper unless it adds genuine value
- **Mandate**: Avoid trying to out-scale Google's 2026 Nature deep RL architecture; focus on interpretable, statistically provable decision-theoretic control.

### 65. Add an explicit prior-art matrix to documentation
- **Mandate**: Maintain `docs/prior_art_matrix.md` rigorously comparing this work to AlphaQubit, Google Nature 2026, Bhardwaj et al. (PRX Quantum 2026), DGR, LCD, and GreenPeas.

### 66. Use the existing negative result as Figure 1 motivation
- **Mandate**: Feature the $27.74\%$ degradation of naive adaptation in Figure 1 to illustrate the fundamental hazard of unconstrained adaptation.

### 67. Do a formal component ablation study
- **Mandate**: Systematically disable components: $- \text{confidence}$, $- \text{abstention (stay)}$, $- \text{switching cost}$, $- \text{hysteresis}$, $- \text{disagreement}$, $- \text{rollback}$.

### 68. Format the complete ablation table
- **Mandate**: Present a 7-column ablation table showing how each algorithmic mechanism progressively suppresses false adaptations and LER.

### 69. Build a synthetic "hidden environment" benchmark
- **Mandate**: Abstract benchmark execution into `Environment(latent_noise_trajectory, observable_stream)` ensuring the evaluator verifies oracle decisions without leaking ground truth.

### 70. Test observability levels explicitly
- **Mandate**: Evaluate controller performance under $O_0$ (syndrome only), $O_1$ (syndrome + daily calibration), $O_2$ (syndrome + calibration + historical statistics), and $O_{\text{oracle}}$.

### 71. Add delayed-feedback experiments
- **Mandate**: Introduce observation lags of 1, 2, 5, and 10 blocks to measure controller resilience under cloud communication latencies.

### 72. Create the drift-to-controller timescale ratio ($\rho = \tau_{\text{adapt}} / \tau_{\text{drift}}$)
- **Mandate**: Quantify system dynamics along $\rho \ll 1$ (effective tracking), $\rho \approx 1$ (marginal), and $\rho \gg 1$ (chasing noise).

### 73. Build the Adaptation Feasibility Map
- **Mandate**: Plot $\tau_{\text{adapt}} / \tau_{\text{drift}}$ vs expected benefit, demarcating the theoretical boundary where adaptation ceases to be viable.

### 74. Standardize the repository modular structure
- **Mandate**: Maintain clean separation across `codes/`, `decoders/`, `estimation/`, `control/`, `noise/`, `hardware/`, and `analysis/`.

### 75. Scientific validity tests must be separated from unit tests
- **Mandate**: Create `tests/scientific/` checking invariance to unobservable labels, paired trajectory seeding, and absence of synthetic fallback.

### 76. Add adversarial stress tests
- **Mandate**: Subject the controller to deceptive scenarios: localized detector hotspots, rate shifts without LER shifts, and rapid oscillatory noise.

### 77. Build a "do nothing" benchmark (Stationary Regime)
- **Mandate**: Verify that under stationary noise, the controller achieves zero false switches, zero rollback, and zero added latency overhead.

### 78. Build an "obvious adaptation" benchmark (Severe Anisotropy)
- **Mandate**: Verify that under persistent asymmetric noise, the controller promptly identifies the discrepancy, switches to reweighted MWPM, and maintains superior LER.

### 79. Build a "too-fast-to-adapt" benchmark (Fast Jitter)
- **Mandate**: Demonstrate that when noise oscillates faster than the estimation window, the controller correctly abstains (`STAY`) rather than thrashing.

### 80. Create a "switching trap" benchmark (Alternating Regimes)
- **Mandate**: Construct alternating regimes where marginal gains are smaller than switching costs; verify that CG-CCADS stays put while naive controllers suffer severe switching penalties.

### 81. Create a "hidden transition" benchmark
- **Mandate**: Test subtle, low-amplitude drift where the controller deliberately delays switching until evidence surpasses confidence thresholds.

### 82. Create a "recovery" benchmark (Post-Burst Normalization)
- **Mandate**: Verify that after a transient error burst clears, the controller safely returns to baseline operation without getting trapped in high-overhead configurations.

### 83. Create a "persistent leakage" benchmark
- **Mandate**: Benchmark performance against stateful, multi-round leakage faults with spatial spreading across neighboring stabilizer checks.

### 84. Hardware should validate core principles, not every simulation permutation
- **Mandate**: Target IBM QPU runs on validating the observation boundary, genuine drift tracking, and paired offline counterfactual decoding on real quantum noise.

### 85. Use the same hardware data for offline counterfactual decoder comparisons
- **Mandate**: Replay raw physical bitstrings through MWPM, RW-MWPM, and UF to compare decoders on identical quantum trajectories.

### 86. Run true interleaved hardware controller experiments
- **Mandate**: Execute randomized $A/B/A/B$ hardware blocks to validate dynamic adaptation against static baselines without temporal confounding.

### 87. Establish a no-synthetic-fallback hardware protocol
- **Mandate**: Mandate that hardware runs lacking complete job IDs, calibration snapshots, or raw bitstrings are flagged as invalid rather than patched with simulation.

### 88. Redesign the result JSON schema around scientific provenance
- **Mandate**: Structure result files with complete software hashes, git commits, configuration digests, raw data hashes, and environment locks.

### 89. Generate every paper number automatically
- **Mandate**: Automate manuscript table and figure generation via scripts (`python -m adaptive_qec.analysis.make_paper_tables`) directly from result artifacts.

### 90. Add a paper.lock file
- **Mandate**: Pin primary experiment IDs, Git commits, artifact hashes, and primary endpoints in `paper.lock` to prevent manuscript drift.

### 91. Pre-register the primary endpoint
- **Mandate**: Declare the primary confirmatory metric: Trajectory-paired delta logical error rate $\bar{\Delta} = \frac{1}{N} \sum_{i=1}^N (\text{LER}_{\text{selective}, i} - \text{LER}_{\text{static}, i})$.

### 92. Pre-register confirmatory vs exploratory regimes
- **Mandate**: Pre-register Stationary, Slow Drift, and Medium Drift as confirmatory; designate Leakage, Bursts, and Hardware runs as exploratory.

### 93. Perform statistical power analysis before final benchmark runs
- **Mandate**: Compute required sample sizes and trajectory counts using pilot variance estimates to achieve $\ge 80\%$ statistical power at $\alpha = 0.05$.

### 94. Report the current 50k result exactly as a historical baseline
- **Mandate**: Transparently document the initial $27.74\%$ failure in the manuscript as the empirical motivation for confidence gating.

### 95. Never hide unfavorable scenarios
- **Mandate**: Dedicate a distinct manuscript section to "Failure Modes and Operating Limits," analyzing exactly where adaptation fails.

### 96. Build the theory around a simple adaptation criterion
- **Mandate**: Formalize the foundational condition: $\text{LCB}(\Delta L_t) > C_{\text{switch}} + C_{\text{latency}} + C_{\text{uncertainty}}$.

### 97. Derive the false-switch probability bound
- **Mandate**: Derive an analytical upper bound on the probability of a harmful switch:
  $$P(\text{Harmful Switch}) \le \Phi\left( -\frac{C_{\text{switch}}}{\sigma_{\Delta}} - z_{1-\alpha} \right)$$

### 98. Derive the analytical adaptation boundary
- **Mandate**: Solve for the critical drift amplitude threshold $A_{\text{drift}}^*$ required to justify policy modification as a function of estimator variance and switching cost.

### 99. Formulate an empirical adaptation regret bound
- **Mandate**: Formalize empirical regret across piecewise-stationary environments with $S$ change points: $R_T \le \mathcal{O}(S \sqrt{T}) + \mathcal{O}(S \cdot C_{\text{switch}})$.

### 100. The central algorithm three-gate architecture
- **Mandate**: Structure the decision pipeline into three hierarchical gates:
  - Gate 1: Drift Evidence ($P(\text{drift}) > \theta_{\text{drift}}$ via CUSUM/SPRT)
  - Gate 2: Policy Relevance ($C_t > \theta_{\text{disagree}}$ via Decoder Disagreement)
  - Gate 3: Cost/Confidence Check ($\text{LCB}(\Delta) > C_{\text{switch}} + C_{\text{latency}}$)

### 101. Do not claim "real-time" without sub-microsecond proof
- **Mandate**: Accurately describe cloud/software orchestration as "macro-timescale adaptive QEC" ($\sim$seconds per batch) and reserve "real-time" for hardware FPGA feedforward ($< 1\,\mu\text{s}$).

### 102. Handle GreenPeas overlap explicitly
- **Mandate**: Cite GreenPeas (JIT hypergraph compilation for adaptive circuits) and delineate: our work operates at the policy decision layer, above decoder compilation.

### 103. Handle Local Clustering Decoder (LCD) overlap explicitly
- **Mandate**: Cite LCD (Nat. Commun. 2025) and clarify: LCD addresses hardware-level runtime graph updates; our work addresses the meta-decision of when and whether to invoke adaptations.

### 104. Implement Bhardwaj et al. as a baseline
- **Mandate**: Integrate a sliding-window syndrome-only noise estimator (Bhardwaj et al., PRX Quantum 2026) as Baseline `B3` in the benchmark hierarchy.

### 105. Treat Google Nature 2026 as motivation, not a target to replicate
- **Mandate**: Complement Google's continuous RL physical control on Willow by addressing the higher-level decision problem: cost-aware selective decoder adaptation with abstention.

### 106. Structure the manuscript narrative around the adaptation paradox
- **Mandate**: Frame the paper around the paradox: *Noise drifts $\to$ adaptation seems necessary $\to$ but adaptation incurs switching, latency, and noise-chasing penalties $\to$ therefore, unconstrained adaptation is often harmful $\to$ selective, confidence-gated adaptation resolves the paradox.*

### 107. Define four clear, defensible paper contributions
- **Mandate**:
  1. A formal decision-theoretic framework for selective QEC adaptation (CG-CCADS) with explicit `STAY`, `SWITCH`, and `ROLLBACK` actions.
  2. An observable-only controller operating without privileged noise states or unobservable LER signals.
  3. A trajectory-paired counterfactual benchmarking methodology evaluating non-stationary QEC under explicit adaptation costs.
  4. An empirically validated 2D adaptation phase diagram mapping the viability boundary of QEC adaptation.

### 108. Explicitly eliminate forbidden hype claims
- **Mandate**: Remove all unsubstantiated claims: *"first adaptive QEC system"*, *"statistically indisputable"*, *"sub-1.2 us real-time hardware control"*, and *"proven superior"*.

### 109. Maintain strictly defensible scientific claims
- **Mandate**: Assert only supported conclusions: conservative confidence-gated switching reduces unnecessary interventions, recovers available adaptation gains, and prevents degradation in noisy regimes.

### 110. Design the final eight paper figures
- **Figure 1**: System Architecture & The 3-Gate Pipeline.
- **Figure 2**: The Adaptation Paradox (Motivating 50k Naive Failure vs Static MWPM).
- **Figure 3**: 2D Adaptation Phase Diagram (Drift Rate vs Amplitude vs Cost).
- **Figure 4**: Performance vs Switching Budget & Adaptation Tax Pareto Frontier.
- **Figure 5**: Full Component Ablation (Naive $\to$ +Stay $\to$ +Cost $\to$ +Confidence $\to$ Full).
- **Figure 6**: Observability Gap (Oracle vs Fully Observable vs Syndrome-Only).
- **Figure 7**: Code Distance Scaling ($d = 3, 5, 7$) and Error Suppression ($\Lambda$).
- **Figure 8**: Interleaved Hardware Execution on IBM Heavy-Hex QPU.

### 111. Construct the seven core manuscript tables
- **Table 1**: Benchmark Noise Suite & Physical Drift Parameters.
- **Table 2**: Baseline Decoder Ladder Properties.
- **Table 3**: Controller Component Specifications.
- **Table 4**: Trajectory-Level Paired Statistical Results.
- **Table 5**: IBM Hardware Backend Metadata & Calibration Telemetry.
- **Table 6**: Latency, Throughput, and Computational Overhead Breakdown.
- **Table 7**: Comparative Prior-Art Demarcation Matrix.

### 112. Execute the 14-phase implementation sequence
- **Sequence**: Phase 0 (Preservation) $\to$ Phase 1 (Integrity & Schemas) $\to$ Phase 2 (Observation Boundary) $\to$ Phase 3 (Geometry & Stabilizer Maps) $\to$ Phase 4 (Baseline Ladder) $\to$ Phase 5 (Counterfactual Engine) $\to$ Phase 6 (CG-CCADS Controller) $\to$ Phase 7 (Trajectory Benchmarking) $\to$ Phase 8 (Stateful Leakage) $\to$ Phase 9 (Predictive Layer) $\to$ Phase 10 (DD Interleaving) $\to$ Phase 11 (Hardware Interleaving) $\to$ Phase 12 (Statistical Analysis) $\to$ Phase 13 (Automated Paper Build).

### 113. Enforce a clean, traceable Git commit sequence
- **Mandate**: Structure commits logically (e.g., `001-archive-v1-snapshot` to `027-preregistered-final-evaluation`) reflecting deliberate scientific progress.

### 114. Reposition CUSUM / SPRT as change detectors
- **Mandate**: Do not treat CUSUM/SPRT as the adaptive algorithm itself; position them as Gate 1 change-point triggers feeding the downstream policy evaluator.

### 115. Refactor the bandit into a candidate policy proposer
- **Mandate**: Confine Exp3/DASE to estimating candidate policy performance within Gate 2, subordinating arm selection to Gate 3 confidence constraints.

### 116. Refactor Dynamical Decoupling as physical characterization
- **Mandate**: Position DD as a physical baseline comparison; report pulse overhead and cycle elongation alongside coherence metrics.

### 117. Reclassify the [[4,2,2]] code hardware results
- **Mandate**: Re-label the [[4,2,2]] IBM experiment as an exploratory demonstration of DD pulse damage ($17.5\% \to 48.8\%$ defect increase), removing claims of fault-tolerant memory.

### 118. Reclassify the distance-3 repetition code experiment
- **Mandate**: Retitle the repetition code hardware result as a smoke test of Qiskit Runtime session execution, noting sequential confounding limitations.

### 119. Maintain the retracted $d=5$ result as a historical case study
- **Mandate**: Document the retracted $d=5$ synthetic result as a prime case study demonstrating why observation boundary enforcement and multi-seed sweeps are essential.

### 120. Finalize publication targeting strategy
- **Target**: Submit to *IEEE Transactions on Quantum Engineering* (IEEE TQE) or *Quantum Science and Technology* (QST) as primary targets; prepare manuscript standards to allow submission to *PRX Quantum*.

### 121. Working paper title
- **Primary Title**: *"When Should Quantum Error Correction Adapt? Confidence- and Cost-Constrained Decoder Selection under Non-Stationary Noise"*
- **Alternative**: *"Selective Adaptive Quantum Error Correction under Drift, Uncertainty, and Switching Costs"*

### 122. Formal paper abstract narrative
- **Sequence**: *Non-stationary noise in physical QPUs motivates adaptive QEC $\to$ However, real-time adaptation incurs switching latency, estimation noise, and risk of policy flapping $\to$ We demonstrate that naive adaptive policies can perform $27.7\%$ worse than static MWPM $\to$ We formulate QEC adaptation as a cost-bearing decision problem under partial observability $\to$ We introduce CG-CCADS featuring confidence gating, decoder disagreement, and rollback $\to$ Validated across 12 noise regimes, code distances $d \in \{3, 5, 7\}$, and IBM Heron hardware $\to$ Our selective framework suppresses unnecessary switches by $80\%+$ while recovering adaptive benefits.*

### 123. The deepest scientific contribution
- **Mandate**: The fundamental pivot from reactive control (*"Observe $\to$ Adapt"*) to decision-theoretic control (*"Observe $\to$ Estimate Uncertainty $\to$ Weigh Switching Costs $\to$ Decide Adapt vs Abstain vs Rollback"*).

### 124. The complete $12 \times 6$ experimental simulation matrix
- **Matrix**: 12 Scenarios (Stationary to Mixed Drift+Leakage) across 6 Controllers (Static MWPM, RW-MWPM, UF, Naive Adaptive, Selective CG-CCADS, Oracle), each evaluated over $\ge 20$ independent trajectories with distance sweeps.

### 125. The comprehensive evaluation metric matrix
- **Metrics**: LER, 95% Wilson CI, paired $\Delta \text{LER}$, effect size (Cohen's $d$), switch count, false-switch rate, rollback frequency, adaptation latency, estimation RMSE, decoder throughput, and circuit depth overhead.

### 126. What NOT to do (Avoid the "Novelty Soup" trap)
- **Mandate**: Do not pile on unrelated buzzwords (deep RL, transformers, continuous gate tuning, qLDPC) to simulate novelty. Focus on methodological depth, statistical rigor, and the selective adaptation decision problem.

### 127. Final end-to-end system architecture
- **Architecture**: Ingest QPU/Simulated Syndromes $\to$ Observable Feature Extraction $\to$ Multiscale Estimator (EWMA/SPRT) $\to$ Candidate Decoder Disagreement Engine $\to$ Confidence & Cost Decision Gate $\to$ Action Execution (`STAY` / `SWITCH` / `ROLLBACK`) $\to$ Closed-Loop Telemetry Logging.

### 128. Definition of "Done" (Submission Readiness Checklist)
- **Checklist**:
  - [x] All 298 existing test cases passing
  - [ ] Claims ledger strictly synchronized with zero unverified claims
  - [ ] Complete elimination of silent synthetic hardware fallback
  - [ ] End-to-end verified observation boundary (no latent state leakage)
  - [ ] True cluster-growth Union-Find benchmarked against PyMatching
  - [ ] Paired counterfactual decoding engine implemented
  - [ ] $\ge 20$ independent trajectories per scenario with frozen seeds
  - [ ] Pre-registered primary endpoint and statistical power analysis
  - [ ] Stateful phenomenological leakage model implemented
  - [ ] Full component ablation table populated
  - [ ] Automated paper figures and tables reproducible via single command

### 129. The three non-negotiable submission gates
- **Gate A (Scientific Correctness)**: Zero oracle peeking, zero fake leakage, zero silent fallback, and fully reconciled claims ledger.
- **Gate B (Algorithmic Novelty)**: Controller demonstrably executes `STAY`, `SWITCH`, and `ROLLBACK` governed by statistical confidence and switching costs.
- **Gate C (Empirical Evidence)**: Validated across $\ge 20$ independent trajectories, distance scaling ($d=3, 5, 7$), held-out seeds, and interleaved QPU hardware execution.
