# Scientific rebuild and publication assessment

**19 November 2026.** This is a real computational QEC research project. It now supports a narrow empirical claim about choosing the time of a decoder update and a reproducible negative result about the practical size of that benefit. I would circulate it to a professor as a research manuscript. I would hold a submission that advertises a superior real-QPU adaptive stack.

## What changed and how it was checked

| Finding in the original audit | Change | Evidence after correction |
|---|---|---|
| An adaptive arm learned from the window it scored, while its competitor had to wait | Every deployable arm starts on the factory graph and acts only after a window ends. A separate known-noise reference is explicit. | First-window parity with factory and no final redundant updates were tested. Per-window errors and actions are released. |
| Some scenarios failed to implement the labeled physical change | Inject the channel inside repeated circuit ticks; select actual data qubits from the circuit; randomize onset, severity, duration and location across seeds. | At distances 3, 5 and 7, tests check channel placement and changes in the detector error model. The experiments model transient between-shot changes in an abstract lattice. |
| Controller checkpoints could restore the wrong matching graph | Removed shadowed methods and preserved the active graph through serialization and reset. | Round-trip test includes the fitted topology and complete edge attributes. |
| Raw short-window reweighting was an easy comparator to beat | Added shrinkage, pooled windows, a capped periodic comparator, historical gate, an independently trained fixed schedule, and matched-noise diagnostics. | At 200 shots raw reweighting made 14,081 primary-panel hotspot errors. Shrinkage made 6,461. Robust regularization accounts for much of the improvement. |
| Multiple tests, degenerate effects and averaging obscured the claim | Froze the original protocol, analyzed paired trajectories, corrected across regimes, reported both p-values and practical size, used strict JSON and recorded count denominators. | The original primary contrast failed. Thirty stationary selective-versus-factory pairs tied. This cannot establish zero risk on all future trajectories. |
| The original comparison used equal caps but unequal realized fits | First evaluated retention and timing post hoc. Froze a separate follow-up in commit bdfc293e before opening 135 new trajectories. Both policies fit exactly once and retain the graph for one window. | At d=3 the alarm won 16 of the 18 trajectories for which its timing differed from the trained calendar. The other 72 used identical timing. |
| Hardware replay used synthetic shots and original IBM scoring mixed observables | Preserved the synthetic artifact with an explicit label. The physical replay now requires ordered registers and compares the predicted Stim observable with its true label. Majority-only scoring is separate. | Regression covers an injected terminal bit flip. Archived IBM job identifiers survive, but their raw registers were unavailable. Read-only retrieval failed on the saved accounts due to IBM account/instance authentication. |
| Original physical mitigation test had weak controls | Future acquisition stores transpiled circuits and register records; uses balanced randomization of paired jobs with matched idle lengths and native X echoes. | The paired construction and durations were checked on a local IBM backend fixture. There was no new QPU submission. This tests bit-flip repetition memory, a different experiment from the surface-code paper. |
| No independent physical surface-code comparison | Bundled released Google detector bits, measured labels, native circuit and publisher predictions. Scored both bases with the same held-out rows. | Local correlated matching and fixed-prior Harmony outperform the selective policy on both public records. Physical acquisition chronology is unknown. |
| The paper timed sleeps and implied a device deadline | Measured decoding, monitoring, graph fit and cached restoration on real arrays. Three warmups and 30 timed trials per case. Used independent data when constructing an already-active graph. | At d=3 and W=2,000 median whole-window host times were 3.329 ms stay, 6.007 ms fit and 3.979 ms reset. Monitoring itself costs about 2.1 ms. Host timings do not establish a device feedback speedup. |
| Claims changed between manuscript, README, UI and artifacts | Rebuilt the manuscript, ledger, source manifest, figure, tables, and paper lock. Removed projected decoder ranks, fake hardware gaps and simulated mitigation verdicts from the dashboard. | The web interface displays the same saved panel counts, runs real same-shot simulator comparisons, and serves the stored calibration as a dated snapshot. Earlier study documents remain archived. |

## Numerical conclusions

The main study has 2,784,000 distinct scored simulation shots. A new, independently seeded confirmation adds 1,620,000. The 88,000 scored public shots belong to two Google records. Re-scoring them with another decoder or window adds no independent measurements.

| Test | Selective | Comparator | Interpretation |
|---|---:|---:|---|
| Original primary, d=3 hotspot, 30 seeds | 6,406 | 6,460 periodic, 360,000 shots | −0.0150 percentage points, 95% CI [−0.0337,+0.0037], Holm p=0.337. Thirty versus sixty fits. |
| Post hoc single fit retained to end, development seeds | 6,437 alarm entry | 6,855 midpoint, same shots | The timing mattered even under a common retention rule. The midpoint was an easy fixed schedule on this onset distribution. |
| New primary, 90 fresh d=3 seeds | 20,025 | 20,164 trained fixed time, 1,080,000 shots | −0.01287 points, 95% CI [−0.02167,−0.00407], p=0.00460. Both used exactly one fit per seed. |
| New descriptive transfer, 45 fresh d=5 seeds | 5,257 | 5,313 trained fixed time, 540,000 shots | −0.01037 points. Direction reproduced. Distance seven was outside this confirmation protocol. |
| Google X, 44,000 scored shots, W=2,000 | 4,680 | 4,555 ordinary shrinkage; 3,889 released fixed-prior Harmony | The selective policy loses to both. |
| Google Z, 44,000 scored shots, W=2,000 | 4,039 | 3,906 ordinary shrinkage; 3,089 released fixed-prior Harmony | Same ordering. One patch, two bases, unknown time ordering. |

On the new d=3 panel 139 errors prevented in 1,080,000 shots is about one fewer logical error per 7,770 memory shots, or 0.69% of the calendar's logical errors. The follow-up interval excludes the 0.060-percentage-point minimum the protocol set. A statistically detectable benefit is real evidence. Presenting it as practically large under the project's own criterion would be misleading.

## Scores and decision

| Dimension | /10 | Reviewer judgment |
|---|---:|---|
| Research importance | 8 | Deciding when to spend a finite calibration budget is experimentally meaningful. |
| Engineering and reproducibility | 8 | Causal code, raw public observations, tested register mapping, seeds, source digests, and replayable counts. Cross-processor Stim sampling and host timing remain environment dependent. |
| Fairness of the timing experiment | 8 | Equal observed budgets and retention, trained fixed baseline and untouched seeds in the follow-up. The hotspot distribution is narrow. |
| Evidence for the narrow timing effect | 7 | Statistically resolved at d=3, descriptively consistent at d=5. The size falls below the chosen minimum. |
| Algorithmic originality | 3 | Weight estimation, conditional graph updates and drift-aware decoding all have direct prior art. |
| Originality of the benchmark question | 6 | A controlled comparison of update timing plus two independent freeze points is a defensible contribution. Worldwide priority has not been established. |
| Experimental hardware evidence for this specific advantage | 3 | The public measurements are authentic. They do not reproduce the simulated advantage over stronger decoders or demonstrate closed-loop operation. |
| Latency evidence | 6 for host profiling | Measures real code paths. End-to-end QPU timing and total system power remain unmeasured. |
| Workshop, reproducibility or empirical benchmark manuscript | 6 | Credible with a narrow framing, honest effect size and the adverse physical comparison. |
| Broad journal claim of a new superior real-time decoder | 2 | Neither algorithmic novelty, physical superiority nor a feedback deadline is established. |

Scores express review judgment, not acceptance probabilities. The remaining scientific question is transfer: whether a costed, one-fit alarm policy predicts better update times across repeated, timestamped physical regimes when compared against a trained calendar and a competitive correlation-aware decoder. The implementation-review document identifies specific competitors and independent data. More shots from the same small synthetic family alone cannot answer that question.

[Manuscript](../papers/main.pdf) · [Results and inference](../VALIDATION.md) · [Original protocol](../PREREG.md) · [Timing protocol](../TIMING_PROTOCOL.md) · [Recent papers and released code](../docs/implementation_review.md) · [Historical audit](publication_audit_2026-11-19/REPORT.md)

## Release verification

The full suite passed 527 tests. Independent reruns reproduced the primary simulation, the single-fit ablation, both phases of the trained-calendar follow-up, Google replay scores and the external decoder comparison. The publication ledger and artifact digests were checked, the manuscript was rendered, and the dashboard was exercised in headless Chrome. A separate third-party IBM Fez check scored terminal majority voting on 8,192 device shots spread across two preparations in each of two bases, with ten parallel chains per shot. NVIDIA's pinned sample writer preserved every detector and label bit in both public Google test splits. Neither integration check is a trained neural decoder benchmark. [External IBM scores](../data/external_ibm_majority.json) · [Upstream code and data](../docs/implementation_review.md)
