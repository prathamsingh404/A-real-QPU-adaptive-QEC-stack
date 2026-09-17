# AdaptiveQEC: A Real-QPU Adaptive Quantum Error Correction Stack

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-113%2F113%20passed%20(100%25)-brightgreen.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Stim](https://img.shields.io/badge/Stim-1.15+-blueviolet.svg)](https://github.com/quantumlib/Stim)
[![PyMatching](https://img.shields.io/badge/PyMatching-2.2+-orange.svg)](https://github.com/oscarhiggott/PyMatching)

AdaptiveQEC is a production-grade, hardware-aware, adaptive Quantum Error Correction (QEC) stack engineered to bridge the fundamental gap between low-level superconducting transmon physics and high-level fault-tolerant algorithms. Designed specifically for IBM Quantum's 156-qubit Heron revision 2 architecture (`ibm_marrakesh`, heavy-hexagonal coupling map), this platform incorporates real-time drift detection, correlated burst isolation (cosmic rays and quasiparticle poisoning), syndrome-based transmon leakage tracking, selective dynamical decoupling (CPMG/XY4/XY8), almost-linear time Union-Find decoding ($O(N \alpha(N))$), and phenomenological threshold scaling analysis ($\Lambda$ ratio).

---

## 1. Multi-Scale System Architecture (Obsidian Knowledge Graph)

The following interactive graph maps the interdependencies across physical hardware, noise phenomenology, syndrome extraction, statistical inference, low-latency decoding, and closed-loop control:

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'fontSize': '13px', 'fontFamily': 'Fira Code, monospace'}}}%%
graph TD
    classDef hardware fill:#1e1e2e,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4;
    classDef physics fill:#181825,stroke:#f38ba8,stroke-width:2px,color:#cdd6f4;
    classDef qec fill:#1e1e2e,stroke:#a6e3a1,stroke-width:2px,color:#cdd6f4;
    classDef noise fill:#181825,stroke:#fab387,stroke-width:2px,color:#cdd6f4;
    classDef decoder fill:#1e1e2e,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4;
    classDef mitigation fill:#181825,stroke:#94e2d5,stroke-width:2px,color:#cdd6f4;
    classDef analysis fill:#1e1e2e,stroke:#f9e2af,stroke-width:2px,color:#cdd6f4;

    subgraph HW ["1. Physical Hardware & Topology"]
        IBM["IBM Heron r2 (ibm_marrakesh)\n156 Transmons | Degree <= 3"]:::hardware
        HH["Heavy-Hex Coupling Map\n(hex cells + flag edge qubits)"]:::hardware
        Twin["Hardware Digital Twin\n(Per-Qubit T1, T2, Readout, CX)"]:::hardware
        Embedding["Surface Code EmbeddingFinder\n(Greedy BFS / SWAP Distance)"]:::hardware
    end

    subgraph Phys ["2. Transmon Physics & Non-Markovian Noise"]
        FluxNoise["1/f Magnetic Flux & Charge Noise\n(Low-frequency dephasing)"]:::physics
        Cosmic["High-Energy Ionizing Radiation\n(Muon impacts & phonon cascades)"]:::physics
        QP["Quasiparticle Poisoning\n(Cooper pair breaking / T1 decay)"]:::physics
        LeakagePhys["Transmon Anharmonicity & Drive\n(|0>, |1> -> |2> non-computational)"]:::physics
    end

    subgraph QECBlock ["3. Fault-Tolerant Circuit Synthesis"]
        StimCirc["Stim Fault-Tolerant Circuit\n(Rotated Surface Code d=3, 5, 7)"]:::qec
        DEM["Detector Error Model (DEM)\n(Separators ^, Hyperedges)"]:::qec
        SyndromeStream["Real-Time Syndrome Stream\ns in {0, 1}^(R x Nd)"]:::qec
    end

    subgraph NoiseDetect ["4. Real-Time Noise & Correlation Engines"]
        CompositeDrift["CompositeDriftDetector\n(EWMA + CUSUM + Burst)"]:::noise
        BurstDet["Poisson Burst Detector\n(P-value < 10^-3, Spatiotemporal)"]:::noise
        LeakageDet["Syndrome Leakage Detector\n(Lag-1 Autocorrelation R(1) + Streaks)"]:::noise
        RateEst["Leakage & Seepage Rates\n(gamma_L, gamma_S, p_steady)"]:::noise
    end

    subgraph Mitigate ["5. Selective Error Mitigation"]
        DDPlanner["AdaptiveDDPlanner\n(CPMG, XY4, XY8 Sequences)"]:::mitigation
        IdleEst["Circuit Idle Window Profiler\n(Spectator qubit dephasing)"]:::mitigation
        DDDecision["Selective Decision Rule\np_dephase(t_idle) > N_pulse * eps_pulse"]:::mitigation
    end

    subgraph Decoders ["6. Dual Low-Latency Decoders"]
        MWPM["MWPMDecoder (PyMatching)\nO(N^3) Edmonds Blossom Baseline"]:::decoder
        UF["UnionFindDecoder (Delfosse & Nickerson)\nO(N alpha(N)) Cluster Radius Matching"]:::decoder
        BurstAware["decode_burst_aware\n(Defect masking during QP avalanches)"]:::decoder
    end

    subgraph AnalysisBlock ["7. Threshold & Scaling Verification"]
        DistSweep["DistanceSweep Orchestrator\n(d in [3, 5, 7], Shots = 2000+)"]:::analysis
        Threshold["ThresholdAnalyzer\nLambda = p_L(d) / p_L(d+2)"]:::analysis
        WilsonCI["Wilson Score 95% Confidence Intervals"]:::analysis
        Fit["Phenomenological Fit\np_L = A * (p_phys / p_th)^((d+1)/2)"]:::analysis
    end

    %% Cross-domain edges
    IBM --> HH
    HH --> Embedding
    Embedding --> StimCirc
    Twin --> StimCirc
    Twin --> DDPlanner

    FluxNoise --> Twin
    Cosmic --> QP
    QP --> BurstDet
    LeakagePhys --> LeakageDet

    StimCirc --> DEM
    StimCirc --> SyndromeStream

    SyndromeStream --> CompositeDrift
    SyndromeStream --> BurstDet
    SyndromeStream --> LeakageDet
    LeakageDet --> RateEst
    RateEst --> Twin

    StimCirc --> IdleEst
    IdleEst --> DDDecision
    DDDecision --> DDPlanner
    DDPlanner --> StimCirc

    DEM --> MWPM
    DEM --> UF
    SyndromeStream --> MWPM
    SyndromeStream --> UF
    BurstDet --> BurstAware
    BurstAware --> MWPM

    MWPM --> DistSweep
    UF --> DistSweep
    DistSweep --> Threshold
    Threshold --> WilsonCI
    Threshold --> Fit
```

---

## 2. The Six Core Engineering Problems

### Problem 1: Heavy-Hex ↔ Surface Code Embedding & SWAP Overhead
* **Hardware Reality**: Planar and rotated surface codes natively require a 4-regular square lattice. IBM Heron r2 processors implement a heavy-hexagonal lattice with vertex degrees $\le 3$.
* **Engineering Solution**: `adaptive_qec.topology.heavy_hex.HeavyHexTopology` and `adaptive_qec.topology.embedding.EmbeddingFinder`. We implement shortest-path routing, unit-cell identification, and greedy BFS embedding to map logical patches onto physical transmons, computing SWAP counts, circuit depth expansion, and spectator idle times.

### Problem 2: Correlated Error Burst Detection (Cosmic Rays & Phonon Avalanches)
* **Physics Context**: When high-energy ionizing radiation (cosmic ray muons, substrate trace radioactivity) strikes the silicon substrate, it deposits mega-electronvolts of energy. This phonon avalanche breaks superconducting Cooper pairs into quasiparticles, temporarily collapsing $T_1$ coherence across dozens of physical transmons simultaneously.
* **Detection Formalism**:
  $$\text{Poisson Test}: \quad P(k \ge K \mid \lambda = w \cdot N_d \cdot p_0) = 1 - \sum_{i=0}^{K-1} \frac{\lambda^i e^{-\lambda}}{i!}$$
