# Comprehensive Research, Architecture & Engineering Journal
## Real-QPU Adaptive Quantum Error Correction Stack

---

## 1. Executive Summary & Core Mission

### 1.1 The User's Vision & Research Philosophy
The goal of this project transcends standard academic exercises or superficial textbook tutorials. It aims directly at the core frontier of quantum information science and engineering:
- **Operating beneath high-level abstractions**: Interrogating the physics of superconducting transmon hardware, understanding non-Markovian noise processes, physical leakage mechanisms, spatiotemporal error correlations, and physical control limits.
- **Architecting real-world QEC pipelines**: Building an end-to-end, hardware-aware, adaptive QEC stack capable of running on IBM Quantum's Heron r2 architecture (`ibm_marrakesh`, 156 qubits) and producing publishable, industrially valuable insights.
- **Bridging the hardware-software gap**: Combining low-latency decoding algorithms ($O(N \alpha(N))$ Union-Find, MWPM), real-time drift detection (EWMA/CUSUM), correlated burst detection (cosmic rays/quasiparticle avalanches), and dynamical decoupling mitigation (XY4/CPMG) into an integrated feedback control system.
- **Commitment to Technical Excellence**: Reaching a level of technical depth where one can work on genuinely difficult, high-impact problems across quantum computing, hardware, HPC, and AI, contributing to research that pushes the field forward.

### 1.2 Hardware Context: IBM Heron r2 (`ibm_marrakesh`)
- **Processor architecture**: Heron revision 2, 156 superconducting transmon qubits arranged in a **heavy-hexagonal lattice**.
- **Coupling constraints**: Average vertex degree $\approx 2.1$, maximum degree 3. Qubits are categorized into data/code vertices and intermediate edge/coupler qubits.
- **Calibration baselines**:
  - Median $T_1 \approx 188.5\ \mu\text{s}$, median $T_2 \approx 130.4\ \mu\text{s}$
  - Single-qubit gate error ($SX, X$) $\approx 2.4 \times 10^{-4}$
  - Two-qubit gate error (ECR / native two-qubit pulse) $\approx 3.02 \times 10^{-3}$
  - Readout error $\approx 1.2 \times 10^{-2}$
- **Primary hardware mismatch**: Topological surface codes require a 2D square grid connectivity graph with degree-4 data qubits and degree-4 ancillas. Mapping this onto heavy-hex requires bridge routing, SWAP insertion, or fold-unfold embeddings that introduce idle-time windows where noise accumulates.

---

## 2. Deep-Dive: The 6 Core Research & Engineering Problems

### Problem 1: Heavy-Hex ↔ Surface Code Mismatch & Embedding
* **Why it matters**: A planar/rotated surface code of distance $d$ requires $d^2$ data qubits and $d^2-1$ measurement ancillas connected in a 4-regular square lattice. The heavy-hex lattice of IBM Heron r2 has max degree 3 and consists of hexagonal tiles with couplers on the edges.
* **The Physics & Overhead**: Direct routing necessitates SWAP networks. Every SWAP gate is composed of 3 CNOT/ECR operations, tripling two-qubit noise along routing channels and introducing significant idle intervals for stationary spectator qubits.
* **Mathematical Lattice Formulation**:
  A heavy-hex lattice consists of vertices $V$ and edges $E$ where nodes alternate in vertical connectivity:
  $$\text{connect\_down}(r, c) = (r \equiv 0 \pmod 2 \land c \equiv 0 \pmod 4) \lor (r \equiv 1 \pmod 2 \land c \equiv 2 \pmod 4)$$
  This guarantees $\max(\deg(v)) \le 3$, with over 50% of vertices having degree 2 (coupler/flag transmons).
* **What We Built**:
  - `src/adaptive_qec/topology/heavy_hex.py`: Graph-theoretic representation of IBM coupling maps, BFS-based shortest path routing, SWAP distance metrics, degree distribution, and diameter analysis.
  - `src/adaptive_qec/topology/embedding.py`: `EmbeddingFinder` and `SurfaceCodeEmbedding` protocol. Implements greedy BFS and candidate-scoring algorithms to embed logical surface codes of arbitrary distance onto physical heavy-hex qubits, computing SWAP counts, depth overhead, and connectivity deficits.
  - `src/adaptive_qec/qec/codes.py`: Enhanced `SurfaceCode.generate_circuit` to accept an optional `embedding` parameter, adjusting the Stim circuit and injecting SWAP noise overhead.

---

### Problem 2: Correlated Error Burst Detection (Cosmic Rays & Quasiparticle Poisoning)
* **Why it matters**: Standard fault-tolerant QEC theory assumes independent, identically distributed (i.i.d.) Pauli errors. Experiments by Google Quantum AI (Nature 2025, Willow processor) and IBM show that high-energy ionizing radiation (cosmic ray muons, environmental radioactivity) causes localized energy absorption in the substrate. This breaks superconducting Cooper pairs, generating cascades of quasiparticles and phonon avalanches that temporarily degrade $T_1$ across dozens of qubits simultaneously.
* **The Detection Theory**:
