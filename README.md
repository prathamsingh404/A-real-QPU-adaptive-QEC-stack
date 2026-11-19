# When should a surface-code decoder rebuild its matching graph?

We compared syndrome-triggered updates with periodic updates under the same two-rebuild cap. On the primary test, triggering saved half the rebuilds and made 54 fewer logical errors in 360,000 shots. The difference was smaller than the predefined useful improvement, and the adjusted p-value was 0.337. On measured Google memory data, ordinary shrinkage beat triggering in both bases at 2,000 shots per window. The contribution is a reproducible experiment measuring when finite-sample updates help and what they cost. The data support a negative primary result.

An independently frozen follow-up tested **exactly one fit and identical graph retention** against a calendar trained on prior trajectories. On 90 fresh distance-three hotspot seeds, the alarm made 20,025 errors versus 20,164 in 1,080,000 shots (paired p = 0.00460). The reduction of 0.01287 percentage points was smaller than our 0.060-point practical threshold. A descriptive distance-five panel agreed in direction. This is evidence of a small timing benefit in a specified simulation. [Follow-up protocol](TIMING_PROTOCOL.md) · [Rebuild report](reports/REBUILD_REPORT.md)

A third frozen comparison tested selective updates with correlated matching. On 768,000 primary simulation shots, selecting a precompiled hotspot model saved 190 logical errors relative to the same library on a trained schedule. Its 0.02474-percentage-point improvement fell below the 0.060-point target. On 384,000 Willow shots the selector used the factory graph throughout, and four released decoders made fewer errors. An additional forward test on 800,000 physical shots across eight ordered acquisitions per basis found 1,855 excess errors versus factory. [Full comparison and cost](reports/CORRELATED_STUDY.md) · [Frozen design](reports/correlated_protocol.md)

[Manuscript](papers/main.pdf) · [All results](VALIDATION.md) · [Internal protocol](PREREG.md) · [Comparison with previous work](docs/prior_art_matrix.md) · [NVIDIA, IBM, Google and recent implementations](docs/implementation_review.md)

## What was compared

Seven deployable policies decode the same observations before updating the graph for the next window. A privileged eighth reference uses the known simulated noise law. The benchmark covers four regimes, three distances, both memory bases at distance three, and two window lengths. Distinct trajectories supply the independent replicates. The larger distances and X basis are secondary panels. The 200- and 2,000-shot windows share shots. Seed-level actions and errors allow the inference to be recalculated.

The original offline external check uses 100,000 public physical shots from two Google records. The first 6,000 in each record calibrate the monitor. The remaining 88,000 score every policy, the publisher's correlated predictions, and an exploratory implementation of DGR's alignment stage. File order is available, acquisition order is unverified. Host profiles measure elapsed CPU-side wall time per window and omit device transport and collection. [Public data attribution](data/public_replay/ATTRIBUTION.md)

In a post hoc sensitivity check, all 30 primary hotspot trajectories used exactly one selective fit. A single fixed midpoint fit made 6,855 errors if retained through recovery, or 6,618 when retained for one window. Moving the fit to alarm entry gave 6,437 and 6,493 respectively. The tested selective policy made 6,406 with alarm-based restoration. This supports investigating the timing of a single update under transient noise. The planned two-fit comparison remains the primary endpoint. [Sensitivity data](data/midpoint_results.json)

## Reproduce locally

Run from the repository root in a Python 3.11 environment. The recorded simulation used Stim 1.16.0, PyMatching 2.4.0, NumPy 2.4.6 and SciPy 1.17.1. Exact Stim sampling can vary across processor instruction sets or versions. The manifest records the original environment and protocol commit.

```sh
python -m pip install -e ".[dev]"
python scripts/verify_lock.py
python scripts/make_claims.py --check
python -m pytest -q
python scripts/reproduce.py
```

The last command independently runs the full simulation in a temporary directory, compares the scientific digest and replays the released physical shots. On a machine with Tectonic installed, `make paper` rebuilds the PDF and plots. For the correlated follow-up, run `python scripts/reproduce_correlated_study.py` to regenerate the development choices, errors and actions, then `python scripts/analyze_correlated_study.py --check` to verify its report and score the released decoder predictions. Run `python scripts/replay_ordered_acquisitions.py --check` and repeat with `--diagnostic --check` for the ordered hardware extension. Host microbenchmarks can be repeated with `python scripts/profile_reweighting.py`; timing samples vary across runs and this command replaces the stored local timing artifact. Start the local results explorer with `uvicorn adaptive_qec.api.app:app`. These research workflows run without QPU access.

## Files for review

| Question | Evidence |
|---|---|
| What was fixed before evaluation? | [Protocol](PREREG.md), [configuration](configs/suite.yaml), [source digests](data/manifest.json) |
| What did each decoder see and when? | [Causal experiment](src/adaptive_qec/experiments/causal_reweighting.py), [per-window data](data/suite_results.json) |
| What were the statistics and observed costs? | [Results ledger](VALIDATION.md), [paired analysis](data/analysis.json), [host traces](data/host_timings.json) |
| How did the policies perform on physical shots? | [Physical replay](data/public_replay_results.json), [alignment-only replay](data/public_alignment_results.json) |
| What happened to the earlier IBM and simulation claims? | [Original audit](reports/publication_audit_2026-11-19/REPORT.md), [superseded artifacts](archive/study_v2/NOTICE.md) |

The IBM repetition-code acquisition and register decoder now support recoverable, block-randomized future jobs. Historical summary files lack the required shot-level registers. The research manuscript uses the public Google experiments as its only physical performance evidence.

To repeat the optional external integration checks, run `python scripts/inspect_public_ibm.py PATH_TO_ibm_fez.tar.gz` on the [Fez archive](https://zenodo.org/records/20768087) and `python scripts/check_ising_sample_format.py PATH_TO_stim_sample_io.py` on [NVIDIA's pinned file](https://github.com/NVIDIA/Ising-Decoding/blob/26305d7e90428761bbe552caaa8676b9bdc44b3c/code/qec/surface_code/stim_sample_io.py). The scripts verify the input digest.
