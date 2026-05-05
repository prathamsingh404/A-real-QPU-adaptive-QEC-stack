# Executive Summary of All 129 Scientific Audit Points & Architectural Reforms
## A Real-QPU Adaptive QEC Stack $\to$ Safe & Selective Adaptive QEC (CG-CCADS)

> **Document Type**: Executive Synthesis & Thematic Digest  
> **Repository**: `github.com/prathamsingh404/A-real-QPU-adaptive-QEC-stack`  
> **Status**: Verified Scientific Roadmap  
> **Core Premise**: Transforming the motivating negative result ($27.74\%$ LER degradation from naive adaptation) into a publishable paper by formulating adaptation as an evidence-based decision problem with switching costs and deliberate abstention.

---

## Thematic Cluster Breakdown of All 129 Points

```
                                  ALL 129 AUDIT POINTS
                                           │
    ┌──────────────────────┬───────────────┴───────────────┬──────────────────────┐
    │                      │                               │                      │
Section 1              Section 2                       Section 3              Section 4
Scientific Integrity   Controller Architecture         Decoders & Baselines   Physical Noise & DD
(Points 1-5, 87-90)    (Points 6-9, 12-25, 60-63, 100) (Points 10-11, 28-29) (Points 30-36, 116-118)
    │                      │                               │                      │
Section 5              Section 6                       Section 7              Section 8
Latency & Reality      Statistics & Suites             Theory & Cost Models   Publishing & Strategy
(Points 38-39, 71-73)  (Points 45-51, 69-83, 91-95)    (Points 27, 43-44, 57) (Points 64-68, 105-129)
```

---

### Cluster 1: Scientific Integrity, Artifacts & Provenance (Points 1–5, 87–90)
* **Point 1 (Preserve v1)**: Tag git revision `v1-audit-historical` and isolate old JSONs to prevent rewriting historical traces.
* **Point 2 (Claims Ledger)**: Transform `VALIDATION.md` into a machine-enforced ledger linking claims to artifact digests.
* **Point 3 (Unified Schemas)**: Mandate `StandardExperimentResult` and `ExperimentManifest` across all runner scripts.
* **Point 4 (Immutable Artifacts)**: Save experiments to self-contained directories with SHA-256 digests for code, configs, and raw outputs.
* **Point 5 (No Silent Fallback)**: In `qiskit_loop.py`, raise `HardwareExecutionError` on QPU failure; never silently generate synthetic Stim data.
* **Point 87 (Hardware Protocol)**: Require valid job IDs, calibration snapshots, and raw bitstrings for all physical data.
* **Point 88 (Provenance JSON)**: Structure all JSON outputs around full software environments, hardware timestamps, and circuit hashes.
* **Point 89 (Automated Extraction)**: Build scripts to generate all manuscript numbers, tables, and plots directly from data files.
* **Point 90 (Paper Lock)**: Implement `paper.lock` pinning experiment hashes and commit revisions to prevent manuscript drift.

---

### Cluster 2: Observation Boundaries, Geometry & Abstractions (Points 6–9, 37)
* **Point 6 (Strict Boundary)**: Provide the controller strictly with `ObservationFrame` (syndrome counts, defect rates, calibration age); block all private simulation states.
* **Point 7 (Boundary Testing)**: Verify via automated tests that shuffling hidden noise labels does not alter controller actions.
* **Point 8 (True Stabilizer Maps)**: Replace naive array bisections (`:n//2`) with `CodeGeometry` mapping detector IDs to $(x, y, t)$ coordinates and $X/Z$ types.
* **Point 9 (Decouple Interventions)**: Separate classical decoder selection (counterfactually evaluable) from physical circuit changes (requiring randomized A/B trials).
* **Point 37 (Observability Table)**: Publish an explicit matrix delineating physically readable telemetry from data allowed into the controller.

---

### Cluster 3: Decoders, Baselines & Counterfactual Evaluation (Points 10–11, 28–29, 85, 102–104)
* **Point 10 (Baseline Ladder)**: Evaluate across 7 hierarchical baselines: Factory MWPM (B0), Optimized MWPM (B1), True UF (B2), Online RW-MWPM (B3), Naive Adaptive (B4), Selective CG-CCADS (B5), Oracle (B6).
* **Point 11 (Genuine Union-Find)**: Verify `src/adaptive_qec/decoders/union_find.py` implements true cluster growth and boundary peeling (Delfosse & Nickerson 2021).
* **Point 28 (Counterfactual Engine)**: Ingest a single syndrome stream into multiple parallel decoders to achieve zero-variance causal comparison.
* **Point 29 (Intervention Separation)**: Formally separate decoder-only counterfactual evaluation from physical circuit modifications.
* **Point 85 (Hardware Replay)**: Replay raw IBM Heron bitstrings through MWPM, RW-MWPM, and UF offline.
* **Point 102 (GreenPeas Delineation)**: Delineate that GreenPeas optimizes JIT hypergraph compilation; our work operates at the policy decision layer.
* **Point 103 (LCD Delineation)**: Clarify that Local Clustering Decoder (Nat. Commun. 2025) implements runtime graph updates; our work determines *when* to adapt.
* **Point 104 (Bhardwaj et al. Baseline)**: Implement Bhardwaj et al. (PRX Quantum 2026) sliding-window syndrome estimation as Baseline B3.

---

### Cluster 4: The CG-CCADS Controller Architecture (Points 12–25, 58–63, 100, 114–115)
* **Point 12 (Core Algorithm)**: Introduce Confidence-Gated Cost-Constrained Adaptive Decoder Selection (CG-CCADS).
* **Point 13 (State Vector)**: Formalize state vector $x_t = [\hat{p}_X, \hat{p}_Z, \hat{p}_{2q}, \hat{r}_{\text{det}}, D_t, C_t, U_t, G_t]$.
* **Point 14 (Multi-Timescale Estimator)**: Couple short-term EWMA anomaly detection, medium-term CUSUM/SPRT change detection, and long-term sliding-window estimation.
* **Point 15 (Decoder Disagreement)**: Compute pairwise discordance $C_t$ across candidate decoders; abstain if decoders agree despite noise drift.
* **Point 16 (Explicit STAY Action)**: Formally implement `STAY` (abstain) with $C_{\text{switch}} = 0$; allow `STAY` to win in stable regimes.
* **Point 17 (Deployable Rewards)**: Remove true LER from online control rewards; use observable risk signals during active operation.
* **Point 18 (Surrogate Objective)**: Formulate objective $R_t = -(\alpha D_t + \beta C_t + \gamma U_t)$ with calibration models.
* **Point 19 (Confidence Bounds)**: Require lower confidence bound $\text{LCB}(\Delta) = \hat{\Delta} - z_{1-\alpha}\sigma_{\Delta} > C_{\text{switch}} + C_{\text{latency}}$.
* **Point 20 (Constrained Optimization)**: Minimize estimated error subject to explicit switching budgets and risk bounds, avoiding arbitrary linear penalty sums.
* **Point 21 (Hysteresis)**: Implement dual thresholds ($T_{\uparrow} > T_{\downarrow}$) to eliminate high-frequency policy flapping.
* **Point 22 (Automatic Rollback)**: Detect post-switch degradation and immediately revert to the previous stable baseline.
* **Point 23 (Probation Period)**: Hold new policies in a probationary state for $K$ blocks before committing long-term.
* **Point 24 (Dead Zone)**: Force `STAY` when marginal performance improvements fall below the statistical noise floor ($|\hat{\Delta}| < \epsilon$).
* **Point 25 (Pseudocode)**: Publish full, reviewable CG-CCADS controller pseudocode in the manuscript.
* **Point 58 (Switching Budget)**: Introduce `AdaptationBudget` limiting switches per hour and added latency.
* **Point 59 (Adaptation Tax)**: Model total adaptation overhead $\tau_{\text{adapt}} = C_{\text{switch}} + C_{\text{latency}} + C_{\text{compute}} + C_{\text{physical}}$.
* **Point 60 (Bandit Integration)**: Subordinate Exp3/DASE bandit logic to proposing candidate policies inside Gate 2, bounded by Gate 3 confidence.
* **Point 61 (Contextual Features)**: Ingest drift velocity, spatial defect asymmetry, and syndrome autocorrelation as context.
* **Point 62 (Strategy Comparison)**: Directly compare Always-Adapt vs Threshold-Adapt vs Selective CG-CCADS.
* **Point 63 (Predictive Extension)**: Implement optional EWMA/AR(1) drift forecasting to evaluate anticipation under sinusoidal noise.
* **Point 100 (Three-Gate Architecture)**: Formalize the hierarchical pipeline: Gate 1 (Drift) $\to$ Gate 2 (Relevance) $\to$ Gate 3 (Cost/Confidence).
* **Point 114 (Reposition CUSUM/SPRT)**: Position CUSUM/SPRT strictly as change-point detection triggers, not adaptive QEC algorithms.
* **Point 115 (Re-scope Bandits)**: Confine bandits to candidate performance estimation rather than unconstrained action selection.

---

### Cluster 5: Physical Noise, Leakage & Dynamical Decoupling (Points 30–36, 116–118)
* **Point 30 (Physical Leakage Model)**: Replace artificial detector-bit injection with stateful phenomenological leakage (persistence, two-qubit propagation, heralding).
* **Point 31 (Leakage Propagation)**: Model CZ/CNOT-mediated fault spreading and neighboring stabilizer excitation.
* **Point 32 (Scope Discipline)**: Keep leakage and DD as evaluation stress tests, not central claims to avoid the "novelty soup" trap.
* **Point 33 (Rebuild DD Experiments)**: Evaluate Free Evolution, XY4, XY8, and CPMG under randomized block assignments.
* **Point 34 (Decouple Detection Rate)**: Decouple defect rate $R_{\text{det}}$ from logical fidelity; explain that DD pulse damage can spike detector activity ($17.5\% \to 48.8\%$).
* **Point 35 (Hardware Interleaving)**: Run randomized blocks ($A/B/B/A$) to isolate control effects from temporal QPU drift.
* **Point 36 (Calibration Telemetry)**: Ingest full IBM Heron calibration snapshots ($T_1, T_2$, gate errors, readout errors) for each block.
* **Point 116 (DD Overhead Reporting)**: Report pulse count, sequence duration, and cycle elongation alongside error suppression metrics.
* **Point 117 (Reclassify [[4,2,2]])**: Treat [[4,2,2]] code data as an exploratory demonstration of DD pulse damage.
* **Point 118 (Reclassify Repetition Code)**: Treat repetition code hardware runs as a runtime infrastructure smoke test.

---

### Cluster 6: Latency, Real-Time Systems & Timescales (Points 38–39, 71–73, 101)
* **Point 38 (Full Latency Profiling)**: Measure and report $p50, p90, p99$ latency for syndrome acquisition, decoding, decision, and communication.
* **Point 39 (Two Architectures)**: Honestly distinguish macro-timescale software control (seconds) from microsecond FPGA feedforward.
* **Point 71 (Delayed Feedback)**: Benchmark controller stability under observation delays of 1, 2, 5, and 10 blocks.
* **Point 72 (Timescale Ratio $\rho$)**: Characterize system dynamics along $\rho = \tau_{\text{adapt}} / \tau_{\text{drift}}$ ($\rho \ll 1$ tracking vs $\rho \gg 1$ chasing noise).
* **Point 73 (Feasibility Map)**: Plot $\tau_{\text{adapt}} / \tau_{\text{drift}}$ vs net adaptation gain to define the physical feasibility frontier.
* **Point 101 (No Unmeasured Claims)**: Eliminate claims of "sub-1.2 us real-time adaptive QEC" for cloud-based Python software loops.

---

### Cluster 7: Statistical Methodology & Benchmark Suite (Points 42, 45–51, 69–70, 75–83, 91–95, 124–125)
* **Point 42 (Analytical Oracle)**: Use Oracle control strictly as an analytical upper bound on achievable adaptation gain.
* **Point 45 (Statistical Unit)**: Establish the independent noise trajectory as the primary statistical unit rather than correlated individual shots.
* **Point 46 (High Trajectory Sample)**: Base all primary claims on $\ge 20$ (target 30) independent trajectories per scenario.
* **Point 47 (Paired Trajectories)**: Run competing controllers on the exact same noise trajectory instance to compute paired differences $\Delta_i$.
* **Point 48 (Seed Partitioning)**: Enforce frozen train (0–9), validation (10–19), and held-out test (20–49) seed splits.
* **Point 49 (12-Scenario Suite)**: Pre-register 12 benchmark regimes covering stationary, drift, bursts, correlations, and leakage.
* **Point 50 (Sub-Threshold Operation)**: Calibrate physical error rates below threshold ($\sim 0.7\% - 1.0\%$) for meaningful distance scaling.
* **Point 51 (Distance Scaling)**: Sweep surface code distances $d \in \{3, 5, 7\}$, reporting error suppression factor $\Lambda$.
* **Point 69 (Hidden Environment)**: Encapsulate benchmark execution in `Environment(latent_trajectory, observable_stream)`.
* **Point 70 (Observability Sweeps)**: Benchmark performance across $O_0$ (syndrome), $O_1$ (syndrome + daily calibration), $O_2$ (syndrome + calibration + historical statistics), and $O_{\text{oracle}}$.
* **Point 75 (Scientific Tests)**: Build dedicated `tests/scientific/` testing invariance to unobservables, absence of fallback, and seed integrity.
* **Point 76 (Adversarial Testing)**: Subject controller to deceptive noise, local hotspots, and rapid jitter.
* **Point 77 (Do-Nothing Benchmark)**: Prove the controller achieves zero false switches in stationary noise.
* **Point 78 (Obvious Adaptation Benchmark)**: Prove the controller promptly switches and sustains lower LER under strong anisotropy.
* **Point 79 (Too-Fast-to-Adapt Benchmark)**: Prove the controller abstains (`STAY`) when noise fluctuates faster than the estimator window.
* **Point 80 (Switching Trap Benchmark)**: Demonstrate that CG-CCADS avoids thrashing in alternating regimes where gains are smaller than switching costs.
* **Point 81 (Hidden Transition Benchmark)**: Measure evidence accumulation lag under low-amplitude drift.
* **Point 82 (Recovery Benchmark)**: Verify safe return to baseline post-burst.
* **Point 83 (Persistent Leakage Benchmark)**: Measure performance against multi-round stateful leakage.
* **Point 91 (Primary Endpoint)**: Pre-register trajectory-paired delta LER ($\bar{\Delta}$) as the sole primary confirmatory endpoint.
* **Point 92 (Confirmatory vs Exploratory)**: Pre-register stationary and slow drift as confirmatory; leakage, bursts, and hardware as exploratory.
* **Point 93 (Power Analysis)**: Calculate sample size and trajectory requirements for $\ge 80\%$ statistical power at $\alpha = 0.05$.
* **Point 94 (Preserve 50k Result)**: Transparently feature the $27.74\%$ naive adaptive failure as the motivating problem in the paper.
* **Point 95 (Transparent Limits)**: Dedicate a distinct manuscript section to "Where Adaptation Fails."
* **Point 124 (Full Simulation Matrix)**: Execute the full $12 \times 6$ matrix (12 Scenarios $\times$ 6 Controllers).
* **Point 125 (Comprehensive Metrics)**: Track LER, Wilson CI, paired $\Delta$, effect size, switches, rollbacks, latency, and throughput.

---

### Cluster 8: Theoretical Formulation & Analytical Bounds (Points 27, 43–44, 52–57, 96–99)
* **Point 27 (Adaptation Value)**: Formalize gross adaptation value $V_{\text{adapt}} = L_{\text{current}} - L_{\text{adapted}}$ and net value $V_{\text{net}} = V_{\text{adapt}} - C_{\text{switch}}$.
* **Point 43 (Adaptation Efficiency)**: Quantify $\eta_{\text{adapt}} = G_{\text{observable}} / G_{\text{oracle}}$ (fraction of oracle advantage recovered).
* **Point 44 (Adaptation Regret)**: Track empirical cumulative regret $R_T$ against hindsight optimum across non-stationary trajectories.
* **Point 52 (Comprehensive Metrics Table)**: Report QEC, adaptation, estimation, systems, and hardware metrics.
* **Point 53 (Pareto Frontiers)**: Plot 2D trade-offs: LER vs Latency and LER vs Switching Overhead.
* **Point 54 (Adaptation Phase Diagram)**: Construct 2D phase diagrams over Drift Rate vs Amplitude, mapping `STAY`, `REWEIGHT`, `SWITCH`, and `ROLLBACK`.
* **Point 55 (Decision Accuracy)**: Measure controller classification performance: precision, recall, false-adaptation rate, and missed-benefit rate.
* **Point 56 (Calibration Curves)**: Plot predicted benefit probability vs empirical benefit frequency.
* **Point 57 (Conformal Prediction)**: Implement distribution-free prediction intervals for estimated policy performance.
* **Point 96 (Analytical Criterion)**: Formalize switching condition: $\text{LCB}(\Delta L_t) > C_{\text{switch}} + C_{\text{latency}} + C_{\text{uncertainty}}$.
* **Point 97 (False-Switch Bound)**: Derive analytical bound on harmful switch probability: $P(\text{Harmful Switch}) \le \Phi\left( -\frac{C_{\text{switch}}}{\sigma_{\Delta}} - z_{1-\alpha} \right)$.
* **Point 98 (Critical Drift Boundary)**: Solve for critical drift amplitude $A_{\text{drift}}^*$ required to justify policy modification.
* **Point 99 (Piecewise-Stationary Regret Bound)**: Formulate regret bound across $S$ change points: $R_T \le \mathcal{O}(S \sqrt{T}) + \mathcal{O}(S \cdot C_{\text{switch}})$.

---

### Cluster 9: Manuscript Framing, Ablation & Publishing Strategy (Points 64–68, 84, 86, 105–113, 119–123, 126–129)
* **Point 64 (Omit Deep RL)**: Exclude heavy black-box RL from the core paper; focus on interpretable decision-theoretic control.
* **Point 65 (Prior-Art Matrix)**: Maintain `docs/prior_art_matrix.md` with explicit differentiation from AlphaQubit, Google Nature 2026, Bhardwaj et al., DGR, LCD, and GreenPeas.
* **Point 66 (Figure 1 Motivation)**: Feature the $27.74\%$ degradation of naive adaptation in Figure 1 to illustrate the adaptation hazard.
* **Point 67 (Component Ablation)**: Systematically ablate confidence gating, abstention (`STAY`), switching cost, hysteresis, and rollback.
* **Point 68 (Ablation Table)**: Format the 7-column ablation table showing progressive suppression of false adaptations.
* **Point 84 (Hardware Focus)**: Target IBM QPU runs on observation boundary validation, drift tracking, and paired counterfactual replay.
* **Point 86 (Interleaved QPU Runs)**: Execute randomized $A/B/A/B$ hardware controller runs on IBM Heron.
* **Point 105 (Google Nature 2026 Framing)**: Frame our contribution as complementary higher-level decision control above hardware-level RL steering.
* **Point 106 (Paradox Narrative)**: Structure the manuscript introduction around the fundamental adaptation paradox.
* **Point 107 (Four Core Contributions)**: State the four core contributions in the introduction.
* **Point 108 (Strip Hype Buzzwords)**: Remove all unsupported marketing terms ("first", "statistically indisputable", "sub-1.2 us real-time").
* **Point 109 (Defensible Claims)**: Assert strictly verified statistical claims regarding conservative confidence-gated switching.
* **Point 110 (Eight Final Figures)**: Render the eight formal manuscript figures.
* **Point 111 (Seven Final Tables)**: Populate the seven formal manuscript tables.
* **Point 112 (14-Phase Order)**: Execute the 14 implementation phases sequentially.
* **Point 113 (Clean Git Commits)**: Maintain a clean, traceable commit history reflecting scientific milestones.
* **Point 119 (Preserve d=5 Case Study)**: Document the retracted synthetic $d=5$ result as a case study in why observation boundaries matter.
* **Point 120 (Target Venues)**: Target submission to *IEEE Transactions on Quantum Engineering* (IEEE TQE) or *Quantum Science and Technology* (QST).
* **Point 121 (Paper Title)**: Primary title: *"When Should Quantum Error Correction Adapt? Confidence- and Cost-Constrained Decoder Selection under Non-Stationary Noise"*.
* **Point 122 (Abstract Narrative)**: Format the abstract following the non-stationary paradox sequence.
* **Point 123 (Core Paradigm Shift)**: Emphasize the transition from reactive control (*"Observe $\to$ Adapt"*) to decision-theoretic control (*"Observe $\to$ Estimate Uncertainty $\to$ Weigh Costs $\to$ Decide"*).
* **Point 126 (Reject Novelty Soup)**: Reject the urge to pile on unintegrated quantum buzzwords.
* **Point 127 (Final Architecture)**: Document the complete closed-loop architecture.
* **Point 128 (Definition of Done)**: Enforce the submission readiness checklist.
* **Point 129 (Three Submission Gates)**: Gate A (Scientific Correctness), Gate B (Algorithmic Novelty), Gate C (Empirical Evidence).
