# Scientific Delineation from Frontier Prior Art (2024–2026)

> **Document Type**: Scientific Architecture & Manuscript Positioning  
> **Target Audience**: Journal Reviewers (*IEEE TQE*, *PRX Quantum*, *QST*)  
> **Relevant Submission Gates**: Gate B (Algorithmic Novelty)  
> **Tasks Addressed**: TASK-023 (Point 102), TASK-024 (Point 103), Frontier Literature Alignment

---

## 1. The Scientific Problem Space

A common reviewer objection in quantum error correction is:
> *"Adaptive decoders and machine-learning controllers have already been proposed. What makes this stack fundamentally novel?"*

This document provides explicit mathematical, architectural, and operational demarcations separating our work (**Confidence-Gated Cost-Constrained Adaptive Decoder Selection, CG-CCADS**) from the major frontier publications.

---

## 2. Delineation Matrix

| Project / Literature | Primary Contribution | Operating Layer | Considers Switching / Latency Cost? | Has Explicit `STAY` (Abstention)? | Our Key Delineation |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **AlphaQubit**<br>*(Google DeepMind, Nature 2024)* | Deep neural network (transformer / recurrent) predicting logical corrections from soft analog readout and syndrome history. | Decoder Primitive (Neural Decoding) | ❌ No | ❌ No (Always decodes via NN) | **AlphaQubit is a single decoder primitive.** CG-CCADS is a meta-controller that orchestrates arbitrary decoders (MWPM, UF, Neural) and decides *when* neural complexity is justified vs when fast MWPM or `STAY` is optimal. |
| **Google RL QEC Control**<br>*(Google Quantum AI Willow, Nature 2026)* | Deep reinforcement learning optimizing physical microwave pulse parameters and gate calibrations. | Physical Hardware Calibration | ❌ No (Continuous parameter adjustment) | ❌ No | **Google 2026 intervenes on physical qubit pulses.** CG-CCADS operates at the classical decoding layer, avoiding destructive quantum state perturbation and runtime calibration overhead. |
| **GreenPeas**<br>*(Chao et al., 2024/2025)* | High-performance Just-In-Time (JIT) compiler generating specialized C++/Rust decoding graph kernels from updated DEMs. | JIT Compilation Engine | ⚠️ Only recompilation compute latency | ❌ No (Compiles whenever DEM changes) | **GreenPeas is an execution engine.** CG-CCADS sits *above* GreenPeas at the supervisory policy layer: it decides *whether* a detected noise drift justifies triggering a graph recompilation or whether the controller should abstain. |
| **Local Clustering Decoder (LCD)**<br>*(Higgott & Breuckmann)* | Distributed decoding algorithm restricting Union-Find cluster growth to local spatial neighborhoods. | Local Decoder Algorithm | ❌ No | ❌ No | **LCD is an algorithmic decoding variant.** CG-CCADS treats LCD as a candidate arm in its baseline ladder, evaluating when local clustering outperforms global matching under heterogeneous drift. |
| **Bhardwaj et al.**<br>*(PRX Quantum 2026 / DGR)* | Sliding-window syndrome-based dynamic graph reweighting (DGR) for MWPM. | Graph Reweighting Algorithm | ❌ No (Always reweights periodically) | ❌ No (Continuous reweighting) | **Bhardwaj et al. assumes continuous adaptation is always beneficial.** In our 50k high-statistics benchmark, naive continuous adaptation was **$27.74\%$ worse** than static MWPM. CG-CCADS provides the confidence-gated test to determine *when* to reweight and when to `STAY`. |

---

## 3. Deep-Dive Delineations

### 3.1. Delineation from GreenPeas (TASK-023, Point 102)
GreenPeas focuses on solving the engineering bottleneck of updating decoding graphs at runtime. When a QPU drifts, GreenPeas generates optimized machine code for the new graph. 

However, GreenPeas does not address the **statistical decision problem**:
- *Is the observed syndrome deviation genuine physical drift or transient Poisson fluctuation?*
- *Do the candidate decoders actually disagree on the next block of syndromes?*
- *Does the expected reduction in logical error rate exceed the switching latency and cache invalidation penalty?*

CG-CCADS provides the exact mathematical gating mechanism required before any tool like GreenPeas is invoked:
$$\text{If } \text{LCB}(\Delta) \le C_{\text{switch}} + C_{\text{latency}} + \epsilon \implies \text{STAY (Do Not Recompile)}$$

### 3.2. Delineation from Local Clustering Decoder (TASK-024, Point 103)
The Local Clustering Decoder (LCD) solves the issue of cluster growth scaling by enforcing hard locality limits on Union-Find. While this reduces worst-case decoding time, it increases logical error rate under correlated noise.

CG-CCADS does not compete with LCD on decoding mechanics. Instead, CG-CCADS uses the pairwise disagreement metric:
$$C_t(d_{\text{MWPM}}, d_{\text{LCD}}) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left[\hat{L}_{i, \text{MWPM}} \neq \hat{L}_{i, \text{LCD}}\right]$$
When $C_t \approx 0$, MWPM and LCD agree on almost all corrections. In that regime, the controller can safely select LCD for lower latency without sacrificing fidelity. When $C_t$ surges, the controller switches back to global MWPM.

### 3.3. Delineation from Bhardwaj et al. / DGR (TASK-025, Point 104)
Bhardwaj et al. (PRX Quantum 2026) showed that updating detector error model edge weights using syndrome coincidence rates improves threshold under drifting noise.

Our work adopts this sliding-window estimator as **Baseline $B_3$** (`OnlineReweightedMWPM`). Crucially, our research reveals the motivating failure mode that Bhardwaj et al. did not address:
Under realistic switching costs and finite estimation windows, continuous adaptation suffers from estimation variance and latency tax:
$$\tau_{\text{adapt}} = C_{\text{switch}} + C_{\text{latency}} + C_{\text{compute}} + C_{\text{physical}}$$
CG-CCADS is the first architecture to formulate adaptation as a **constrained confidence-gated decision** with an explicit zero-cost `STAY` action.
