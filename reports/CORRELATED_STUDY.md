# Correlated matching under a two-update cap

Seven policies scored identical shots. Commit `c2f80af` froze the code, choices, public inputs and internal protocol before evaluation. Negative paired differences favor selective timing. A shot is a complete memory experiment with one binary logical label.

## Frozen primary results

Distance three, Z memory, 24 independent trajectories in each of four regimes, 768,000 shots. Pointwise 95% Student intervals and two-sided paired t tests use trajectories as units. Holm adjustment covers both tests. The three success criteria were an adjusted p below .05, a reduction of at least 0.060 percentage points, and aggregate online time at most the matched calendar’s.

| Policy vs calendar | Errors, policy / calendar | Difference in percentage points [95% CI] | Raw p | Holm p | Host time, policy / calendar (s) | Passed all three |
|---|---:|---:|---:|---:|---:|---|
| selective vs calendar | 11,849 / 11,878 | -0.00378 [-0.01102, +0.00346] | 0.30309 | 0.30309 | 2.391 / 2.442 | No |
| library_budget vs library_calendar | 11,386 / 11,576 | -0.02474 [-0.03801, -0.01146] | 0.00036164 | 0.00072327 | 1.428 / 1.461 | No |

## Absolute error counts on every condition

Calendar and always fit full empirical priors. Selective chooses factory, local or full. The three library columns respectively allow all looks, two development-selected looks, or all looks with at most two uses. Model selection uses detector data from the preceding window. Uniform and bias are transfer checks. Physical rows each cover one released record in file order.

| Panel | Shots | factory | calendar | always | selective | library | library_calendar | library_budget |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| d=3, stationary | 192,000 | 2,106 | 2,102 | 2,101 | 2,106 | 2,106 | 2,106 | 2,106 |
| d=3, abrupt | 192,000 | 3,429 | 3,425 | 3,416 | 3,423 | 3,210 | 3,265 | 3,169 |
| d=3, gradual | 192,000 | 3,039 | 3,042 | 3,022 | 3,015 | 2,877 | 2,937 | 2,856 |
| d=3, recurring | 192,000 | 3,311 | 3,309 | 3,326 | 3,305 | 3,238 | 3,268 | 3,255 |
| d=3, uniform | 192,000 | 4,289 | 4,280 | 4,264 | 4,274 | 4,508 | 4,388 | 4,488 |
| d=3, bias | 192,000 | 2,206 | 2,211 | 2,132 | 2,155 | 2,276 | 2,220 | 2,276 |
| d=5, stationary | 64,000 | 354 | 354 | 352 | 354 | 354 | 354 | 354 |
| d=5, abrupt | 64,000 | 451 | 451 | 446 | 450 | 431 | 430 | 425 |
| d=5, gradual | 64,000 | 432 | 434 | 414 | 422 | 420 | 427 | 416 |
| d=5, recurring | 64,000 | 462 | 459 | 458 | 464 | 470 | 457 | 464 |
| d=5, uniform | 64,000 | 1,358 | 1,350 | 1,380 | 1,388 | 1,424 | 1,406 | 1,419 |
| d=5, bias | 64,000 | 447 | 448 | 435 | 439 | 470 | 450 | 469 |
| Willow d3_0+0j, X | 96,000 | 5,966 | 5,974 | 5,985 | 5,966 | 5,966 | 5,966 | 5,966 |
| Willow d3_0+0j, Z | 96,000 | 5,986 | 5,951 | 5,924 | 5,986 | 5,986 | 5,986 | 5,986 |
| Willow d3_0+4j, X | 96,000 | 6,525 | 6,536 | 6,581 | 6,515 | 6,525 | 6,525 | 6,525 |
| Willow d3_0+4j, Z | 96,000 | 7,869 | 7,880 | 7,942 | 7,891 | 7,869 | 7,869 | 7,869 |
| Willow pooled, descriptive | 384,000 | 26,346 | 26,341 | 26,432 | 26,358 | 26,346 | 26,346 | 26,346 |

## Released predictions on the same physical shots

This exploratory comparison reads the publisher’s predictions on exactly the first 96,000 shots of each record. Training and calibration independence from these shots is unverified. Runtime for these decoders was not measured. The sparse-blossom column uses the archive member name. The release README describes a correlated matching pathway under a different name.

| Record | Our budgeted library | Released sparse blossom | Tesseract SI1000 | Tesseract calibrated | AlphaQubit 2 |
|---|---:|---:|---:|---:|---:|
| d3_0+0j, X | 5,966 | 5,696 | 4,932 | 5,023 | 4,568 |
| d3_0+0j, Z | 5,986 | 5,051 | 4,596 | 4,566 | 4,350 |
| d3_0+4j, X | 6,525 | 6,433 | 5,813 | 5,784 | 5,343 |
| d3_0+4j, Z | 7,869 | 6,228 | 5,797 | 4,721 | 4,371 |
| Pooled | 26,346 | 23,408 | 21,138 | 20,094 | 18,632 |

The budgeted library selected a hotspot model for 0 measured windows out of 336 possible next windows.

## Forward transfer between physical acquisitions

Google 72-qubit set 1 provides eight ordered acquisition samples per basis at one distance-three layout. Each decision scores a later sample. The 50 interleaved streams per basis share acquisition conditions. Counts are descriptive. Each column within a row uses the same 400,000 physical shots.

| Basis | factory | calendar | always | selective | library | library_calendar | library_budget | Harmony SI1000 | Harmony optimized |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| X | 26,819 | 26,877 | 27,116 | 26,870 | 27,794 | 26,819 | 27,779 | 23,808 | 21,578 |
| Z | 23,882 | 23,877 | 23,874 | 23,832 | 24,777 | 23,882 | 24,777 | 20,865 | 20,209 |

The common factory circuit is the first sample’s SI1000 model. The sources attribute temporal order at the acquisition level. The design makes no assumption about the chronology of rows within a sample. Both schedules and model strengths were frozen in simulation. The Harmony rows use released predictions and the training independence of optimized priors is unverified. [Exploratory protocol](ordered_acquisition_protocol.md) · [Per-stream counts and actions](../data/correlated_study/ordered.json) · [Source acquisition records](../data/ordered_acquisitions/ATTRIBUTION.md)

### Did the selected model still fit one acquisition later?

Post hoc diagnostic restricted to active budgeted-library windows. We recomputed the label-free marginal composite log likelihood on the deployment window and compared the chosen graph with factory. Positive error differences are excess logical errors under the chosen graph.

| Next-window score vs factory | Active windows | Excess logical errors |
|---|---:|---:|
| higher | 19 | +323 |
| lower or equal | 96 | +1,532 |

This tests persistence of the selection score and its alignment with the logical objective. The grouped results describe these acquisitions. They do not identify the device fault responsible.

## Measured work

The primary panel includes 768 windows per arm. Milliseconds per 1,000 shots are medians across windows. Times cover host decoding and selection or refitting. The library is shared between its three policies. First use may incur lazy initialization. Cold setup below measures graph construction separately. Acquisition, transport, scoring and device feedback lie outside these clocks.

| Policy | Active next windows | Decode ms | Decision/build ms | Total ms, 95th percentile |
|---|---:|---:|---:|---:|
| factory | 0 | 1.604 | 0.002 | 2.807 |
| calendar | 192 | 1.823 | 0.022 | 7.876 |
| always | 672 | 2.012 | 4.252 | 9.521 |
| selective | 128 | 1.767 | 0.481 | 8.284 |
| library | 194 | 1.579 | 0.092 | 3.159 |
| library_calendar | 65 | 1.631 | 0.003 | 3.038 |
| library_budget | 144 | 1.589 | 0.066 | 3.159 |

Simulated distance 3 required 10 graphs and 151.0 ms of cold library construction.

Simulated distance 5 required 26 graphs and 1846.7 ms of cold library construction.

The four physical libraries required 869.5–962.4 ms each.

## What the design can resolve

On 256,000 development shots, a label-informed hindsight selector could have saved at most 103 errors against factory among the tested empirical estimates with two single-window uses. The practical margin would require 154. This ceiling covers the enumerated candidates and traces only. Development selected empirical calendar looks 1 and 7, library looks 2 and 3, and penalty 32 (one-based looks).

The library knows that a candidate fault has the form of depolarization on a data qubit at each circuit tick. The simulator generates the local disturbance from that same family with different strength, timing and location. Improvement under this match is evidence about model selection given a fault family. The physical records cover two neighboring patches, two bases, ten cycles and a single calibration category. Consecutive file segments share a source and have no verified timestamps or drift labels. Their errors are descriptive. A physical timing test requires repeated, chronologically documented episodes and an experimental unit that reflects the acquisition design.

The cap limits active windows per episode. Schedules can inspect different numbers of windows. Cost is an observed criterion on one CPU host. The protocol supplies neither an enforced device deadline nor interleaved hardware acquisition.

[Protocol](correlated_protocol.md) · [Simulation traces](../data/correlated_study/simulation.json) · [Physical traces](../data/correlated_study/physical.json) · [Data attribution](../data/willow_timing/ATTRIBUTION.md)
