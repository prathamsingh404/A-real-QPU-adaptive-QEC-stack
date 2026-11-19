# Scientific Claims Ledger & Reproducibility Matrix

> **Audit Standard**: Zero-tolerance empirical verification, transparent provenance delineation, and strict artifact synchronization.  
> **Last Verification**: October 2026 (Integrity Reset & Rigorous Pipeline Compilation)  
> **Cryptographic Provenance Lock**: `paper.lock` (SHA-256 verified)

This ledger tracks every quantitative claim across documentation, connecting each claim to its exact artifact, reproduction command, and verified status. All confirmatory claims are derived dynamically from `data/suite_results.json` and locked by `paper.lock`.

---

## 1. Active Confirmatory Claims (Current Manuscript)

| Claim ID | Category | Claim Statement | Verified Status | Artifact / Value | Reproduction Command | Notes |
| :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| **CLAIM-CONF-01** | Scaled QEC | Below-Threshold Error Suppression ($\Lambda_{3 \to 5} = 1.543 > 1.0$) | **REPRODUCED** | $\Lambda_{3 \to 5} = 1.543$ | `python -m pytest tests/test_empirical_scaling_and_ablation.py -k test_distance_scaling` | Evaluated on rotated surface codes at $p_{\text{base}} = 0.004 < p_{\text{th}} \approx 0.01$. Confirms below-threshold operation ($d=3$ LER $= 0.00693$, $d=5$ LER $= 0.00449$). |
| **CLAIM-CONF-02** | Adaptation Boundary | Small-Window Variance Explosion at $W=200$ ($+104.38\%$ LER degradation) | **REPRODUCED** | $+104.38\%$ ($\text{LER}: 0.02009 \to 0.04107$) | `python src/adaptive_qec/analysis/make_paper_tables.py` | At $W=200$, finite-sample coincidence noise corrupts Dijkstra edge weights. CG-CCADS safely enforces abstention ($92.7\%$ stay rate, $\text{LER} = 0.02035$). |
| **CLAIM-CONF-03A** | Localized Drift | Statistically Significant LER Reduction under Localized Hotspot Drift | **REPRODUCED** | $\bar{\Delta} = -0.000433$ ($[-0.000768, -0.000092]$, $t = -2.6001$, $p = 0.01454$) | `python src/adaptive_qec/analysis/make_paper_tables.py` | Evaluated across $N=30$ independent held-out trajectories (seeds 1000–1029) at $W=2{,}000, d=3$. Static LER $= 0.025533 \pm 0.001449$, CG-CCADS LER $= 0.025100 \pm 0.001506$ ($-1.70\%$, $32.7\%$ adaptation rate). |
| **CLAIM-CONF-03B** | Negative Control | Safe Zero-Cost Abstention (`STAY`) under Stationary Baseline | **REPRODUCED** | $\bar{\Delta} = +0.000057$ ($[-0.000059, +0.000173]$, $t = 1.0000$, $p = 0.3256$) | `python src/adaptive_qec/analysis/make_paper_tables.py` | Static LER $= 0.011550 \pm 0.001070$, CG-CCADS LER $= 0.011600 \pm 0.01110$. Stay rate $= 100.0\%$. Controller incurs zero false adaptations ($p > 0.05$). |
| **CLAIM-CONF-03C** | Negative Control | Safe Invariance Abstention (`STAY`) under Uniform Global Drift | **REPRODUCED** | $\bar{\Delta} = +0.000157$ ($[-0.000034, +0.000347]$, $t = 1.6738$, $p = 0.1052$) | `python src/adaptive_qec/analysis/make_paper_tables.py` | Static LER $= 0.031977 \pm 0.001784$, CG-CCADS LER $= 0.032133 \pm 0.001712$. Stay rate $= 96.7\%$. Theorem 6.1 invariance holds; no significant difference ($p > 0.05$). |
| **CLAIM-CONF-03D** | Control Scenario | Safe Invariance Abstention (`STAY`) under Structural Anisotropy | **REPRODUCED** | $\bar{\Delta} = +0.000027$ ($[-0.000028, +0.000081]$, $t = 1.0000$, $p = 0.3256$) | `python src/adaptive_qec/analysis/make_paper_tables.py` | Static LER $= 0.011317 \pm 0.001070$, CG-CCADS LER $= 0.011343 \pm 0.001110$. Stay rate $= 100.0\%$. Controller avoids unnecessary switching cost ($p > 0.05$). |
| **CLAIM-CONF-04** | Latency Budget | Sub-Millisecond Host CPU Orchestration Overhead | **REPRODUCED** | $912.51\,\mu\text{s}$ total ($317.51\,\mu\text{s}$ decision) | `python -m pytest tests/test_empirical_scaling_and_ablation.py -k test_latency_budget` | Amortized per-round overhead of $0.16\,\mu\text{s}$ at $W=2000$. Operates completely off-critical-path during hardware shot batch collection. |
| **CLAIM-CONF-05** | Pipeline Integrity | Zero Hardcoded Result Literals in Table and Figure Generators | **REPRODUCED** | 0 AST violations | `python -m pytest tests/test_no_hardcoded_results.py` | Abstract Syntax Tree (AST) checker scans generators, forbidding non-whitelisted numerical literals. All values flow directly from JSON. |
| **CLAIM-CONF-06** | Cryptographic Lock | SHA-256 Provenance Verification Across All Tracked Files | **REPRODUCED** | 11/11 files matched | `python scripts/verify_lock.py` | Cryptographically binds data files, manifest, LaTeX tables, and figures to git commit. |

---

## 2. Granular Confirmatory Experiment Metrics ($N=30$ Held-Out Seeds)

The table below documents the exact empirical distributions obtained from `data/suite_results.json` evaluated across seeds 1000–1029:

| Scenario | Metric | Static Baseline | CG-CCADS (Ours) | Paired $\bar{\Delta}$ | $95\%$ Confidence Interval | $t$-stat | $p$-value | Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Localized Hotspot** | Mean LER | $0.025533$ | $0.025100$ | **$-0.000433$** | $[-0.000768, -0.000092]$ | $-2.6001$ | **$0.01454$** | **ADAPT (Win, Reject $H_0$)** |
| | Std Dev | $0.001449$ | $0.001506$ | — | — | — | — | Stay Rate: $67.3\%$ |
| **Stationary Baseline** | Mean LER | $0.011550$ | $0.011600$ | $+0.000057$ | $[-0.000059, +0.000173]$ | $+1.0000$ | $0.3256$ | **STAY (Accept $H_0$, Safe)** |
| | Std Dev | $0.001070$ | $0.001110$ | — | — | — | — | Stay Rate: $100.0\%$ |
| **Uniform Drift** | Mean LER | $0.031977$ | $0.032133$ | $+0.000157$ | $[-0.000034, +0.000347]$ | $+1.6738$ | $0.1052$ | **STAY (Accept $H_0$, Safe)** |
| | Std Dev | $0.001784$ | $0.001712$ | — | — | — | — | Stay Rate: $96.7\%$ |
| **Structural Anisotropy**| Mean LER | $0.011317$ | $0.011343$ | $+0.000027$ | $[-0.000028, +0.000081]$ | $+1.0000$ | $0.3256$ | **STAY (Accept $H_0$, Safe)** |
| | Std Dev | $0.001070$ | $0.001110$ | — | — | — | — | Stay Rate: $100.0\%$ |

---

## 3. Historical Claims: Audited, Downgraded, and Retracted

| Claim ID | Category | Claim Statement | Verified Status | Expected / Artifact Value | Artifact Path | Notes |
|:---|:---|:---|:---:|:---|:---|:---|
| **CLAIM-HIST-01** | Hardware Benchmark | [[4,2,2]] Code Detection Rate on IBM Heron (ibm_marrakesh) | **DOWNGRADED** | `0.175` | `data/hardware_results/ibm_marrakesh_true_quantum_and_dynamic_results.json` | DISPROVED PRIOR CLAIM: VALIDATION.md previously claimed 'Detection Rate = 100% | PROVEN'. The actual committed artifact records an error detection rate of 17.5% (unmitigated) and 48.8% (DD). Furthermore, under DD, X-stabilizer defects jumped from 83 to 404 (4.8x increase), indicating DD pulse damage. |
| **CLAIM-HIST-02** | Hardware Benchmark | Dynamic Feedforward Syndrome Correction Latency | **DOWNGRADED** | `18` | `data/hardware_results/ibm_marrakesh_true_quantum_and_dynamic_results.json` | DISPROVED PRIOR CLAIM: Prior claim asserted 'Latency < 1.2 us | PROVEN'. The artifact contains zero latency timing measurements. The latency was an unmeasured architectural assumption. |
| **CLAIM-HIST-03** | Hardware Benchmark | Repetition Code Distance-3 Memory on ibm_marrakesh | **DOWNGRADED** | `0.038` | `data/hardware_results/ibm_marrakesh_qec_results.json` | CAUTION: Unmitigated and DD jobs were submitted sequentially, not interleaved ABAB. Bit-flip repetition code is blind to pure dephasing; observed difference may be temporal drift between sequential jobs. |
| **CLAIM-HIST-04** | Decoder Throughput | PyMatching C++ Sparse Blossom Baseline Throughput | **REPRODUCED** | ~300,000 shots/s (CPU) | *Code / Test Suite* | Standard PyMatching v2 library performance on distance-3 Stim circuits. |
| **CLAIM-HIST-05** | Decoder Throughput | Precomputed All-Pairs Shortest Paths Union-Find Throughput | **DOWNGRADED** | ~100,000 shots/s (d=3 only) | *Code / Test Suite* | Limited to distance d=3 with 8 detectors where lookup matrix is trivial. Does not generalize to scaled distance codes without quadratic graph memory. Heuristic Dijkstra merge, not exact Delfosse-Nickerson linear bucket growth. |
| **CLAIM-HIST-06** | Decoder Architecture | Lazy MWPM Hybrid Routing | **DOWNGRADED** | Fallback heuristic implemented in codebase | *Code / Test Suite* | Heuristic fallback; no proof of asymptotic superiority over pure MWPM under physical noise. |
| **CLAIM-HIST-07** | Bandit Controller | DASE Bandit Amnesia-Free Arm Retention & Sublinear Regret | **DOWNGRADED** | Heuristic bandit implementation with fast-path overrides | *Code / Test Suite* | DISPROVED REGRET BOUND CLAIM: Exp3/DASE theoretical regret bounds are mathematically broken by deterministic heuristic fast-paths injected into the action selection loop. |
| **CLAIM-HIST-08** | Error Mitigation | Dynamical Decoupling Coherence Improvement Factor | **DOWNGRADED** | Literature citation (Pokharel et al., PRL 2023) | *Code / Test Suite* | External literature result, not an independently measured baseline in this repository. |
| **CLAIM-HIST-09** | Telemetry | SPRT / CUSUM Drift Detection Sensitivity | **REPRODUCED** | Passes synthetic unit tests | *Code / Test Suite* | Synthetic validation passing; operational tracking on QPU noise still pending closed-loop integration. |
| **CLAIM-HIST-10** | Telemetry | Cosmic Ray Poisson Burst Detection Latency | **REPRODUCED** | Passes synthetic unit tests | *Code / Test Suite* | Synthetic validation passing on simulated Poisson spikes. |
| **CLAIM-HIST-11** | Practical Workload | 3-bit Molecular H2 IQPE Chemical Accuracy | **DISPROVED / RETRACTED** | `False` | `data/hardware_results/ibm_marrakesh_practical_benchmarks_results.json` | DISPROVED PRIOR CLAIM: Prior claim asserted chemical accuracy demonstration. Actual committed artifact has chemical_accuracy_achieved=false with an energy error of 761.5 kcal/mol (over 760x above the 1 kcal/mol threshold for chemical accuracy). Removed from core paper scope. |
| **CLAIM-HIST-12** | Practical Workload | Deterministic Teleportation Fidelity on Heavy-Hex | **DOWNGRADED** | `0.919` | `data/hardware_results/ibm_marrakesh_practical_benchmarks_results.json` | DISPROVED PRIOR CLAIM: Prior claim stated fidelity = 96.2%. Committed artifact shows 91.90% for dynamic feedforward, 93.06% for post-selected, and 97.90% for swap-network. Removed from core QEC paper scope as standard demonstration. |
| **CLAIM-HIST-13** | Simulation Study | Adaptive Advantage at Scaled Distances (d=5, +10.05% Error Reduction) | **NOT REPRODUCED** | `10.05` | *Code / Test Suite* | CRITICAL AUDIT FINDING: Single-seed artifact (seed 47). Multi-seed sweeps yield 12-14% worse LER for adaptive. Furthermore, simulation injected leakage by naively setting detector 2 and 5 to 1 without modifying the logical observable, while controller peeked at schedule.is_leakage_active and invoked a hardcoded fast-path rule. Status: Unreproducible and retracted as a headline scientific claim. |
| **CLAIM-HIST-14** | Simulation Study | 50,000-Shot Headline Adaptive vs Static LER on Distance-3 | **DISPROVED / RETRACTED** | `-27.74` | `experiments/results/adaptive_vs_static_high_stats_50k.json` | CRITICAL AUDIT FINDING: README previously claimed Adaptive LER = 0.130220 vs MWPM 0.170320 (+23.54% win). The actual committed JSON artifact states Static MWPM LER = 0.170320, Adaptive LER = 0.217560, meaning Adaptive was 27.74% WORSE than Static MWPM. The prior documentation completely inverted reality. |
| **CLAIM-HIST-15** | Simulation Study | Simulated IBM Heron Interleaved A/B Benchmark | **RETRACTED AS HARDWARE EVIDENCE** | Synthetic binomial simulation | *Code / Test Suite* | CRITICAL AUDIT FINDING: Previously presented as real hardware interleaved A/B benchmark on IBM Heron (`ibm_heron_marrakesh`). Code investigation revealed it was a synthetic binomial simulation driven by hand-typed numbers. Status: Deleted from hardware directory; moved to `examples/synthetic_ab_demo.py` with explicit `"provenance": "synthetic"` flag; completely removed from all manuscript tables and claims. Zero simulated hardware claims exist. |

---

## 4. Status Definitions
- **REPRODUCED**: Independently verified and matches artifact JSON byte-for-byte or passes automated tests.
- **DOWNGRADED**: Claim was previously inflated or lacked proper controls; rewritten with accurate physical boundaries.
- **DISPROVED / RETRACTED**: Committed data directly contradicts prior claims, or claimed metrics were never measured.
- **NOT REPRODUCED**: Claim was observed on an isolated single seed or broken path; fails multi-seed statistical sweeps.\n