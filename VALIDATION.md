# Scientific Claims Ledger & Reproducibility Matrix

> **Audit Standard**: Zero-tolerance empirical verification, transparent provenance delineation, and strict artifact synchronization.  
> **Last Verification**: October 2026 (Integrity Reset & Rigorous Pipeline Compilation)  
> **Cryptographic Provenance Lock**: `paper.lock` (SHA-256 verified)

This ledger tracks every quantitative claim across documentation, connecting each claim to its exact artifact, reproduction command, and verified status.

---

## 1. Active Confirmatory Claims (Current Manuscript)

| Claim ID | Category | Claim Statement | Verified Status | Artifact / Value | Reproduction Command | Notes |
| :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| **CLAIM-CONF-01** | Scaled QEC | Below-Threshold Error Suppression ($\Lambda_{3 \to 5} = 1.543 > 1.0$) | **REPRODUCED** | $\Lambda_{3 \to 5} = 1.543$ | `python -m pytest tests/test_empirical_scaling_and_ablation.py -k test_distance_scaling` | Evaluated on rotated surface codes at $p_{\text{base}} = 0.004 < p_{\text{th}} \approx 0.01$. Confirms below-threshold operation. |
| **CLAIM-CONF-02** | Adaptation Boundary | Small-Window Variance Explosion at $W=200$ ($+104.38\%$ LER degradation) | **REPRODUCED** | $+104.38\%$ ($\text{LER}: 0.02009 \to 0.04107$) | `python src/adaptive_qec/analysis/make_paper_tables.py` | At $W=200$, finite-sample coincidence noise corrupts Dijkstra edge weights. CG-CCADS safely enforces abstention ($92.7\%$ stay rate). |
| **CLAIM-CONF-03** | Localized Drift | Statistically Significant LER Reduction under Localized Hotspot Drift | **REPRODUCED** | $\bar{\Delta} = -0.0004$ ($[-0.0008, -0.0001]$, $p = 0.0145$) | `python src/adaptive_qec/analysis/make_paper_tables.py` | Evaluated across $N=30$ independent held-out trajectories (seeds 1000–1029) at $W=2{,}000, d=3$ under identical syndrome streams. |
| **CLAIM-CONF-04** | Negative Controls | Safe Zero-Cost Abstention (`STAY`) under Stationary and Uniform Drift | **REPRODUCED** | $p = 0.3256$ (Stationary), $p = 0.1052$ (Uniform) | `python src/adaptive_qec/analysis/make_paper_tables.py` | Trajectory-paired deltas $\bar{\Delta}$ are statistically indistinguishable from zero ($p > 0.05$); controller stays without false switches. |
| **CLAIM-CONF-05** | Pipeline Integrity | Zero Hardcoded Result Literals in Table and Figure Generators | **REPRODUCED** | 0 violations | `python -m pytest tests/test_no_hardcoded_results.py` | Abstract Syntax Tree (AST) checker scans generators, forbidding non-whitelisted numerical literals. |
| **CLAIM-CONF-06** | Cryptographic Lock | SHA-256 Provenance Verification Across All Tracked Files | **REPRODUCED** | 11/11 files matched | `python scripts/verify_lock.py` | Cryptographically binds data files, manifest, LaTeX tables, and figures to git commit. |

---

## 2. Historical Claims: Audited, Downgraded, and Retracted

| Claim ID | Category | Claim Statement | Verified Status | Expected / Artifact Value | Artifact Path | Notes |
|:---|:---|:---|:---:|:---|:---|:---|
| **CLAIM-HIST-01** | Hardware Benchmark | [[4,2,2]] Code Detection Rate on IBM Heron (ibm_marrakesh) | **DOWNGRADED** | `0.175` | `data/hardware_results/ibm_marrakesh_true_quantum_and_dynamic_results.json` | DISPROVED PRIOR CLAIM: VALIDATION.md previously claimed 'Detection Rate = 100% \| PROVEN'. The actual committed artifact records an error detection rate of 17.5% (unmitigated) and 48.8% (DD). Furthermore, under DD, X-stabilizer defects jumped from 83 to 404 (4.8x increase), indicating DD pulse damage. |
| **CLAIM-HIST-02** | Hardware Benchmark | Dynamic Feedforward Syndrome Correction Latency | **DOWNGRADED** | `18` | `data/hardware_results/ibm_marrakesh_true_quantum_and_dynamic_results.json` | DISPROVED PRIOR CLAIM: Prior claim asserted 'Latency < 1.2 us \| PROVEN'. The artifact contains zero latency timing measurements. The latency was an unmeasured architectural assumption. |
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
| **CLAIM-HIST-13** | Simulation Study | Adaptive Advantage at Scaled Distances (d=5, +10.05% Error Reduction) | **NOT REPRODUCED** | `10.05` | `experiments/results/adaptive_vs_static_d5_20260930_225600.json` | CRITICAL AUDIT FINDING: Single-seed artifact (seed 47). Multi-seed sweeps yield 12-14% worse LER for adaptive. Furthermore, simulation injected leakage by naively setting detector 2 and 5 to 1 without modifying the logical observable, while controller peeked at schedule.is_leakage_active and invoked a hardcoded fast-path rule. Status: Unreproducible and retracted as a headline scientific claim. |
| **CLAIM-HIST-14** | Simulation Study | 50,000-Shot Headline Adaptive vs Static LER on Distance-3 | **DISPROVED / RETRACTED** | `-27.74` | `experiments/results/adaptive_vs_static_high_stats_50k.json` | CRITICAL AUDIT FINDING: README previously claimed Adaptive LER = 0.130220 vs MWPM 0.170320 (+23.54% win). The actual committed JSON artifact states Static MWPM LER = 0.170320, Adaptive LER = 0.217560, meaning Adaptive was 27.74% WORSE than Static MWPM. The prior documentation completely inverted reality. |
| **CLAIM-HIST-15** | Simulation Study | Simulated IBM Heron Interleaved A/B Benchmark | **RETRACTED AS HARDWARE EVIDENCE** | Synthetic binomial simulation | `examples/synthetic_ab_demo.py` | CRITICAL AUDIT FINDING: Previously presented as real hardware interleaved A/B benchmark on IBM Heron (`ibm_heron_marrakesh`). Code investigation revealed it was a synthetic binomial simulation driven by hand-typed numbers. Status: Deleted from hardware directory; moved to `examples/synthetic_ab_demo.py` with explicit `"provenance": "synthetic"` flag; completely removed from all manuscript tables and claims. |

---

## Status Definitions
- **REPRODUCED**: Independently verified and matches artifact JSON byte-for-byte or passes automated tests.
- **DOWNGRADED**: Claim was previously inflated or lacked proper controls; rewritten with accurate physical boundaries.
- **DISPROVED / RETRACTED**: Committed data directly contradicts prior claims, or claimed metrics were never measured.
- **NOT REPRODUCED**: Claim was observed on an isolated single seed or broken path; fails multi-seed statistical sweeps.
