# Prior work and contribution assessment

| Source | Method already demonstrated | Relationship to this study |
|---|---|---|
| Spitz et al. (2017 preprint, 2018 journal) | Infers matching weights from syndrome statistics in a changing environment | Our pairwise estimator follows this approach. Factory mixing and update schedules are separate choices. [Paper](https://arxiv.org/abs/1712.02360) |
| Wang et al., DGR (2023–24) | Estimates weights from occurrences of decoded edges, then conditionally adjusts weights for correlations | Our additional public replay independently implements the occurrence-based alignment stage, clips extreme frequencies and updates the next window. The correlation stage and published hardware design are outside this comparison. [Methods, section IV](https://arxiv.org/html/2311.16214v3) |
| Bhardwaj et al. (2026) | Derives sliding-window filtering behavior and tracks drift with iterative and relative-window estimators | Our two-window pool is a simple comparator. It does not reproduce their spectral or relative-window algorithm. A claim of first windowed adaptation would be indefensible. [Paper](https://journals.aps.org/prxquantum/abstract/10.1103/z1hc-nqw5) · [Authors' code](https://github.com/Dev-ph-quan-1203/Estimation-of-Drifting-Noise-in-QEC) |
| Poster et al., ReloQate (2026 preprint) | Uses detector fire rates to predict logical performance and trigger tile remapping for recalibration | Our alarm changes decoder edge weights. Their intervention acts on a different resource. The general idea of triggering on detector telemetry is established. [Paper](https://arxiv.org/abs/2603.00837) |
| Google Quantum AI | Published physical memories, observable flips and correlated-decoder predictions | We score the provided predictions on precisely the file rows scored by our policies. The experimental data and baseline predictions belong to their authors. [Release](https://zenodo.org/records/13273331) |

The defensible contribution is testing an operational question with common shots, a common rebuild cap, strictly delayed updates, independent trajectories and measured host costs. No broad estimator, threshold, optimality or hardware feedback novelty follows. The closest prior methods overlap substantially. This search cannot establish worldwide priority for the exact design.

The [expanded implementation review](implementation_review.md) covers NVIDIA Ising, IBM Relay-BP and Pêlagos, Google and DeepMind, QAdapt v3, physical closed-loop control and recent learned priors.
