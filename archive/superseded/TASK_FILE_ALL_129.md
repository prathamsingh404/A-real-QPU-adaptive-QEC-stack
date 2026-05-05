# TASK FILE: Selective Adaptive QEC (CG-CCADS) - All 129 Actionable Tasks
# Branch: research/selective-adaptation-v2 | Updated: 2026-10-22

## GROUND TRUTH NEGATIVE RESULT (Never Erase)
  static MWPM:    8516/50000 = 0.170320
  naive adaptive: 10878/50000 = 0.217560  (27.74% WORSE)
  Research Question: When is adaptation worth doing, and how can a controller decide to abstain?

---

## THE THREE SUBMISSION GATES
- [x] **Gate A**: Scientific Correctness (No oracle leakage, no fake leakage bits, no synthetic fallback, reconciled claims)
- [x] **Gate B**: Algorithmic Novelty (CG-CCADS demonstrates STAY, SWITCH, and ROLLBACK under explicit costs)
- [x] **Gate C**: Empirical Evidence (>=20 independent trajectories, held-out test seeds, distance scaling, hardware interleaving)

---

## THE 129 ACTIONABLE IMPLEMENTATION TASKS

### Phase 0: Research Preservation & Baseline Freezing
- [x] **TASK-001** [Point 1]: Freeze current project tip and tag `v1-audit-historical` in research snapshot.
- [x] **TASK-002** [Point 94]: Explicitly preserve the 50k benchmark (0.17032 vs 0.21756) as the motivating baseline failure.
- [x] **TASK-003** [Point 117]: Reclassify IBM Heron [[4,2,2]] code result as exploratory study of DD pulse damage.
- [x] **TASK-004** [Point 118]: Reclassify IBM distance-3 repetition code run as runtime infrastructure smoke test.
- [x] **TASK-005** [Point 119]: Preserve retracted d=5 result as case study in oracle avoidance and boundary enforcement.

### Phase 1: Scientific Integrity & Provenance Infrastructure
- [x] **TASK-006** [Point 2]: Completely redesign claims ledger (`VALIDATION.md`) with machine-verifiable schema.
- [x] **TASK-007** [Point 3]: Enforce `StandardExperimentResult` dataclass across all runner scripts.
- [x] **TASK-008** [Point 4]: Make experiment outputs immutable with SHA-256 hashes for circuits, raw data, and metrics.
- [x] **TASK-009** [Point 5]: Eliminate silent synthetic fallback in `qiskit_loop.py`; raise `HardwareExecutionError`.
- [x] **TASK-010** [Point 87]: Enforce hardware execution protocol requiring valid job IDs, calibration, and bitstrings.
- [x] **TASK-011** [Point 88]: Redesign result JSON schema around full software and hardware provenance.
- [x] **TASK-012** [Point 90]: Implement `paper.lock` pinning primary experiments, endpoints, and commit hashes.

### Phase 2: Observation Boundary & Geometry
- [x] **TASK-013** [Point 6]: Construct strict `ObservationFrame` excluding latent noise states, true LER, and schedules.
- [x] **TASK-014** [Point 7]: Implement end-to-end tests verifying controller invariance to unobservable latent permutations.
- [x] **TASK-015** [Point 8]: Replace naive array slicing (`:n//2`) with true `CodeGeometry`, `DetectorMap`, and `StabilizerMap`.
- [x] **TASK-016** [Point 9]: Separate classical decoder adaptation from physical quantum circuit interventions.
- [x] **TASK-017** [Point 37]: Create explicit manuscript table demarcating available telemetry vs allowed controller observations.

### Phase 3: Decoders & Baseline Ladder
- [x] **TASK-018** [Point 10]: Implement complete baseline ladder: Factory MWPM (B0), Optimized MWPM (B1), True UF (B2), Online RW-MWPM (B3), Naive Adaptive (B4), Selective CG-CCADS (B5), Oracle (B6).
- [x] **TASK-019** [Point 11]: Formally verify true cluster-growth Union-Find against standard peeling algorithms.
- [x] **TASK-020** [Point 28]: Implement counterfactual decoding engine feeding identical syndrome streams to all decoders.
- [x] **TASK-021** [Point 29]: Separate decoder-only counterfactual evaluation from randomized A/B circuit interventions.
- [x] **TASK-022** [Point 85]: Replay raw IBM hardware bitstrings offline through all decoders for paired comparison.
- [x] **TASK-023** [Point 102]: Document delineation from GreenPeas (operating at policy level, above JIT graph compilation).
- [x] **TASK-024** [Point 103]: Document delineation from Local Clustering Decoder (focusing on when to adapt, not just how).
- [x] **TASK-025** [Point 104]: Implement Bhardwaj et al. (PRX Quantum 2026) sliding-window estimator as Baseline B3.

### Phase 4: The CG-CCADS Controller Architecture
- [x] **TASK-026** [Point 12]: Implement core Confidence-Gated Cost-Constrained Adaptive Decoder Selection (CG-CCADS).
- [x] **TASK-027** [Point 13]: Formalize mathematical controller state $x_t = [\hat{p}_X, \hat{p}_Z, \hat{p}_{2q}, \hat{r}_{\text{det}}, D_t, C_t, U_t, G_t]$.
- [x] **TASK-028** [Point 14]: Build multi-timescale estimator (EWMA short-term, CUSUM/SPRT medium-term, sliding-window long-term).
- [x] **TASK-029** [Point 15]: Implement decoder disagreement engine computing $C_t$ across candidate decoder pairs.
- [x] **TASK-030** [Point 16]: Add explicit `STAY` (deliberate abstention) as a first-class, zero-cost action.
- [x] **TASK-031** [Point 17]: Replace online $1 - \text{LER}$ reward with deployable observable surrogate metrics.
- [x] **TASK-032** [Point 18]: Formulate surrogate objective $R_t = -(\alpha D_t + \beta C_t + \gamma U_t)$ with calibration models.
- [x] **TASK-033** [Point 19]: Implement conservative lower confidence bound switching rule: $\text{LCB}(\Delta) > C_{\text{switch}} + C_{\text{latency}}$.
- [x] **TASK-034** [Point 20]: Formulate switching decision as constrained optimization rather than arbitrary linear weights.
- [x] **TASK-035** [Point 21]: Implement dual-threshold hysteresis ($T_{\uparrow} > T_{\downarrow}$) to eliminate policy flapping.
- [x] **TASK-036** [Point 22]: Build automatic rollback mechanism when active decoder performance deteriorates.
- [x] **TASK-037** [Point 23]: Implement probationary switching period ($K$ blocks) before policy commitments.
- [x] **TASK-038** [Point 24]: Add dead zone ($|\hat{\Delta}| < \epsilon \implies \text{STAY}$) to ignore statistically indistinguishable gains.
- [x] **TASK-039** [Point 25]: Publish complete, reviewable CG-CCADS controller pseudocode.
- [x] **TASK-040** [Point 58]: Introduce explicit `AdaptationBudget` constraint.
- [x] **TASK-041** [Point 59]: Formulate physical "adaptation tax" model: $\tau_{\text{adapt}} = C_{\text{switch}} + C_{\text{latency}} + C_{\text{compute}} + C_{\text{physical}}$.
- [x] **TASK-042** [Point 60]: Refactor bandit (Exp3/DASE) to propose candidate policies within the confidence-gated pipeline.
- [x] **TASK-043** [Point 61]: Add contextual features (drift velocity, spatial asymmetry, syndrome autocorrelation) to policy selection.
- [x] **TASK-044** [Point 62]: Compare three strategies: Always-Adapt, Threshold-Adapt, and Selective CG-CCADS.
- [x] **TASK-045** [Point 63]: Implement optional predictive selective adaptation layer (EWMA/AR(1) forecasting).
- [x] **TASK-046** [Point 100]: Formalize three-gate pipeline: Gate 1 (Drift), Gate 2 (Relevance/Disagreement), Gate 3 (Cost/Confidence).
- [x] **TASK-047** [Point 114]: Reposition CUSUM/SPRT strictly as change-point detection triggers.
- [x] **TASK-048** [Point 115]: Re-scope bandit algorithms as candidate performance estimators, subordinate to Gate 3.

### Phase 5: Physical Noise, Leakage & Dynamical Decoupling
- [x] **TASK-049** [Point 30]: Replace artificial detector-bit injection with stateful phenomenological leakage model.
- [x] **TASK-050** [Point 31]: Model leakage propagation through two-qubit gates and multi-round persistence.
- [x] **TASK-051** [Point 32]: Keep leakage and DD as stress tests and physical extensions, not central novelty claims.
- [x] **TASK-052** [Point 33]: Rebuild DD benchmark comparing Free Evolution, XY4, XY8, and CPMG under randomized block designs.
- [x] **TASK-053** [Point 34]: Explicitly decouple detection rate $R_{\text{det}}$ from logical fidelity and pulse damage.
- [x] **TASK-054** [Point 35]: Formulate rigorous randomized block hardware protocol ($A/B/B/A$) for physical runs.
- [x] **TASK-055** [Point 36]: Ingest full IBM backend calibration snapshots ($T_1, T_2$, gate errors, readout errors) for each block.
- [x] **TASK-056** [Point 116]: Document DD pulse duration and gate overhead alongside error suppression metrics.

### Phase 6: Latency, Real-Time Systems & Timescales
- [x] **TASK-057** [Point 38]: Profile full latency stack: $T_{\text{meas}} + T_{\text{decode}} + T_{\text{decision}} + T_{\text{comm}}$, reporting $p50, p90, p99$.
- [x] **TASK-058** [Point 39]: Formally distinguish macro-timescale orchestration (seconds) from microsecond FPGA feedforward.
- [x] **TASK-059** [Point 71]: Evaluate controller resilience under delayed feedback ($1, 2, 5, 10$ blocks).
- [x] **TASK-060** [Point 72]: Formalize drift-to-controller timescale ratio $\rho = \tau_{\text{adapt}} / \tau_{\text{drift}}$.
- [x] **TASK-061** [Point 73]: Construct Adaptation Feasibility Map plotting $\tau_{\text{adapt}} / \tau_{\text{drift}}$ vs net benefit.
- [x] **TASK-062** [Point 101]: Eliminate all unsupported "real-time sub-microsecond" claims for software loops.

### Phase 7: Benchmarks, Scenarios & Adversarial Testing
- [x] **TASK-063** [Point 49]: Predefine 12-scenario benchmark suite covering stationary, drift, bursts, correlation, and leakage.
- [x] **TASK-064** [Point 50]: Calibrate all simulation baselines below the fault-tolerant threshold ($\sim 0.7\% - 1.0\%$).
- [x] **TASK-065** [Point 69]: Build synthetic `Environment(latent_trajectory, observable_stream)` abstraction.
- [x] **TASK-066** [Point 70]: Test controller under four observability levels: $O_0, O_1, O_2, O_{\text{oracle}}$.
- [x] **TASK-067** [Point 75]: Build dedicated `tests/scientific/` checking invariance to unobservables and seeding integrity.
- [x] **TASK-068** [Point 76]: Implement adversarial tests (localized hotspots, rate shifts without LER shifts, deceptive noise).
- [x] **TASK-069** [Point 77]: Execute "Do Nothing" benchmark verifying zero false switches in stationary noise.
- [x] **TASK-070** [Point 78]: Execute "Obvious Adaptation" benchmark verifying prompt switching under strong anisotropy.
- [x] **TASK-071** [Point 79]: Execute "Too Fast to Adapt" benchmark verifying abstention under high-frequency noise jitter.
- [x] **TASK-072** [Point 80]: Execute "Switching Trap" benchmark showing selective controller avoids thrashing penalties.
- [x] **TASK-073** [Point 81]: Execute "Hidden Transition" benchmark measuring evidence accumulation delay.
- [x] **TASK-074** [Point 82]: Execute "Recovery" benchmark verifying safe return to baseline post-burst.
- [x] **TASK-075** [Point 83]: Execute "Persistent Leakage" benchmark under stateful spatial spreading.

### Phase 8: Statistical Rigor & Power Analysis
- [x] **TASK-076** [Point 45]: Establish independent noise trajectory as the primary statistical unit (not correlated raw shots).
- [x] **TASK-077** [Point 46]: Execute $\ge 20$ (target 30) independent trajectories per scenario for core study.
- [x] **TASK-078** [Point 47]: Enforce paired trajectory analysis computing $\Delta_i = \text{LER}_{\text{adaptive}, i} - \text{LER}_{\text{static}, i}$.
- [x] **TASK-079** [Point 48]: Implement strict seed partitioning: Train (0–9), Validation (10–19), Test (20–49).
- [x] **TASK-080** [Point 91]: Pre-register primary confirmatory endpoint: trajectory-paired delta LER ($\bar{\Delta}$).
- [x] **TASK-081** [Point 92]: Pre-register confirmatory regimes (stationary, slow drift) vs exploratory regimes.
- [x] **TASK-082** [Point 93]: Conduct formal statistical power analysis ensuring $\ge 80\%$ power at $\alpha = 0.05$.
- [x] **TASK-083** [Point 95]: Include explicit manuscript section detailing failure modes and boundary conditions.

### Phase 9: Theoretical Modeling & Analytical Proofs
- [x] **TASK-084** [Point 27]: Implement gross adaptation value ($V_{\text{adapt}}$) and net adaptation value ($V_{\text{net}}$) metrics.
- [x] **TASK-085** [Point 43]: Quantify adaptation efficiency $\eta_{\text{adapt}} = G_{\text{observable}} / G_{\text{oracle}}$.
- [x] **TASK-086** [Point 44]: Formulate and compute empirical cumulative regret $R_T$ against hindsight optimum.
- [x] **TASK-087** [Point 57]: Implement distribution-free conformal prediction intervals for policy LER estimates.
- [x] **TASK-088** [Point 96]: Formalize foundational switching condition: $\text{LCB}(\Delta L_t) > C_{\text{switch}} + C_{\text{latency}} + C_{\text{uncertainty}}$.
- [x] **TASK-089** [Point 97]: Derive analytical upper bound on false-switching probability.
- [x] **TASK-090** [Point 98]: Derive analytical adaptation boundary threshold $A_{\text{drift}}^*$ as function of noise variance and cost.
- [x] **TASK-091** [Point 99]: Derive regret bound under piecewise-stationary drift with $S$ change points: $\mathcal{O}(S \sqrt{T} + S \cdot C_{\text{switch}})$.

### Phase 10: Empirical Scaling & Ablation Analysis
- [x] **TASK-092** [Point 51]: Execute code distance sweeps $d \in \{3, 5, 7\}$, reporting error suppression factor $\Lambda$.
- [x] **TASK-093** [Point 52]: Generate comprehensive metric table (QEC, adaptation, estimation, systems, hardware metrics).
- [x] **TASK-094** [Point 53]: Generate 2D Pareto frontiers (LER vs Latency, LER vs Switching Overhead).
- [x] **TASK-095** [Point 54]: Construct 2D empirical adaptation phase diagram (Drift Rate vs Amplitude vs Cost).
- [x] **TASK-096** [Point 55]: Benchmark controller decision accuracy (precision, recall, false-adaptation rate).
- [x] **TASK-097** [Point 56]: Generate calibration curves (predicted benefit probability vs observed frequency).
- [x] **TASK-098** [Point 67]: Execute formal component ablation study (Full vs $-$Confidence vs $-$Stay vs $-$Cost vs $-$Rollback).
- [x] **TASK-099** [Point 68]: Format and populate complete 7-column ablation table.

### Phase 11: Real Hardware Validation on IBM Heavy-Hex
- [x] **TASK-100** [Point 84]: Focus QPU hardware validation on observation boundary and interleaved A/B comparisons.
- [x] **TASK-101** [Point 86]: Execute true interleaved $A/B/A/B$ hardware controller runs on IBM Heron.

### Phase 12: Manuscript Framing, Figures & Tables
- [x] **TASK-102** [Point 64]: Exclude black-box deep RL from core paper to maintain clean decision-theoretic focus.
- [x] **TASK-103** [Point 65]: Create and maintain `docs/prior_art_matrix.md` with detailed comparative literature analysis.
- [x] **TASK-104** [Point 66]: Position historical 50k negative result as Figure 1 motivating paradox.
- [x] **TASK-105** [Point 105]: Frame relation to Google Nature 2026 as complementary higher-level decision control.
- [x] **TASK-106** [Point 106]: Structure manuscript introduction around the fundamental adaptation paradox.
- [x] **TASK-107** [Point 107]: Define four core contributions in manuscript introduction.
- [x] **TASK-108** [Point 108]: Strip all forbidden hype buzzwords across entire codebase and documentation.
- [x] **TASK-109** [Point 109]: Frame all paper claims around validated, defensible statistical boundaries.
- [x] **TASK-110** [Point 110]: Render all eight final paper figures using matplotlib/seaborn vector graphics.
- [x] **TASK-111** [Point 111]: Populate all seven formal manuscript tables directly from data artifacts.
- [x] **TASK-112** [Point 120]: Target submission to IEEE TQE or Quantum Science and Technology.
- [x] **TASK-113** [Point 121]: Finalize paper title: *"When Should Quantum Error Correction Adapt? Confidence- and Cost-Constrained Decoder Selection under Non-Stationary Noise"*.
- [x] **TASK-114** [Point 122]: Format abstract narrative following the non-stationary paradox structure.
- [x] **TASK-115** [Point 123]: Emphasize the core paradigm shift: Observe $\to$ Estimate Uncertainty $\to$ Weigh Costs $\to$ Decide.
- [x] **TASK-116** [Point 124]: Complete full $12 \times 6$ simulation matrix across all scenarios and baselines.
- [x] **TASK-117** [Point 125]: Generate complete metric matrix across all trajectory runs.
- [x] **TASK-118** [Point 126]: Maintain strict focus; reject unrelated "novelty soup" features.
- [x] **TASK-119** [Point 127]: Document final end-to-end architecture diagram in manuscript and repo.
- [x] **TASK-120** [Point 128]: Complete and verify submission readiness checklist.

### Phase 13: Automated Paper Build & Reproducibility Suite
- [x] **TASK-121** [Point 74]: Enforce standardized modular directory structure across `src/adaptive_qec/`.
- [x] **TASK-122** [Point 89]: Build `python -m adaptive_qec.analysis.make_paper_tables` and `make_paper_figures`.
- [x] **TASK-123** [Point 112]: Verify all 14 execution phases pass in sequential order.
- [x] **TASK-124** [Point 113]: Verify clean Git commit history reflecting verifiable milestones.
- [x] **TASK-125** [Point 129]: Execute validation test suite for Gate A, Gate B, and Gate C.
- [x] **TASK-126**: Implement CI GitHub Action checking claims ledger against result artifacts on every push.
- [x] **TASK-127**: Package complete open-source Zenodo artifact with Docker container and environment locks.
- [x] **TASK-128**: Generate LaTeX tables (`table_1.tex` through `table_7.tex`) automatically from verified artifacts.
- [x] **TASK-129**: Final audit and sign-off review with academic professor before journal submission.
