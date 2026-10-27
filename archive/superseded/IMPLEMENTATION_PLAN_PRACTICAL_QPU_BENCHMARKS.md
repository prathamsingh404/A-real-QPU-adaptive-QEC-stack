# Implementation Plan: Practical Real-QPU Workloads with Zero Simulation

**Target QPU:** IBM Heron revision 2 (`ibm_marrakesh`, 156 heavy-hexagonal transmons)  
**Execution Paradigm:** Pure physical QPU execution (zero hardcoded values, zero synthetic mock data, verifiable IBM Runtime Job IDs)  
**Modules to Build:**
1. **Part 1:** Molecular Quantum Chemistry via Iterative Quantum Phase Estimation (IQPE) for $H_2$ Ground-State Dissociation Curve.
2. **Part 2:** Deterministic Multi-Hop Quantum Teleportation using On-Chip Dynamic Conditional Feedforward and Idle Dynamical Decoupling.

---

## 1. Executive Context & Scientific Baseline

### Why These Specific Applications?
Reviewers and domain experts rightly dismiss synthetic benchmarks that run repetition codes on artificial noise. To establish genuine scientific utility, an adaptive quantum stack must demonstrate:
1. **Physical Utility:** Solving a real quantum mechanical Hamiltonian (molecular electronic structure) where state preparation and eigenvalue extraction map to real chemistry.
2. **Dynamic Quantum Control:** Executing deterministic quantum state transfer where classical mid-circuit measurement outcomes dictate sub-microsecond quantum gate operations inside the control hardware.

### Uncompromising Scientific Constraints
- **Zero Simulation:** All primary results must originate from real QPU executions submitted through Qiskit Runtime Sampler V2.
- **Zero Hardcoded Numbers:** No pre-baked expectation values, no hardcoded energies, no synthetic plotting arrays. All data must parse live from physical bitstring counts and state tomography.
- **Direct Prior Art Comparison:** We must quantitatively compare our adaptive/mitigated results against the existing standard baselines in literature.

---

## 2. Part 1: Molecular Quantum Chemistry via IQPE ($H_2$ Molecule)

### A. Physical Problem Formulation
We compute the ground-state electronic energy of the Hydrogen molecule ($H_2$) in a minimal STO-3G basis across internuclear bond distances $R \in [0.3\,\text{Å}, 2.5\,\text{Å}]$.

Using the Jordan-Wigner / parity mapping with $\mathbb{Z}_2$ symmetry tapering (spin conservation $S_z=0$ and electron number conservation $N_e=2$), the Hamiltonian reduces to an effective 1-qubit active space Hamiltonian:
$$\hat{H}_{H_2}(R) = g_0(R)\hat{I} + g_1(R)\hat{Z} + g_2(R)\hat{X}$$
where coefficients $g_0(R), g_1(R), g_2(R)$ are analytically computed from nuclear repulsion and one-/two-electron Coulomb integrals:
- $g_0(R) = \frac{1}{R} + h_{00} + h_{11} + \frac{1}{2}h_{0110}$
- $g_1(R) = \frac{1}{2}(h_{00} - h_{11})$
- $g_2(R) = h_{0101}$

The ground state energy is $E_0(R) = g_0(R) - \sqrt{g_1(R)^2 + g_2(R)^2}$.

### B. Algorithm Architecture: Iterative Quantum Phase Estimation (IQPE)
Standard Quantum Phase Estimation (QPE) requires $m$ counting qubits and an inverse QFT network of depth $O(m^2)$, which experiences catastrophic decoherence on current transmons.

**IQPE Architecture:**
- **Qubit Allocation:** Exactly **1 ancilla qubit** ($q_{\text{anc}}$) + **1 system qubit** ($q_{\text{sys}}$).
- **Precision:** $m$ bits of phase precision ($m=3$ or $m=4$, yielding energy precision $\Delta E \sim \frac{2\pi}{2^m \tau}$).
- **Step-by-step iteration (from least significant bit $k=m$ to most significant bit $k=1$):**
  1. Initialize $q_{\text{anc}} \leftarrow |+\rangle$.
  2. Apply controlled unitary $C\text{-}U^{2^{k-1}}$ where $U = e^{-i \hat{H}\tau}$.
  3. **Real-time Feedforward:** If $k < m$, apply phase correction rotation $R_z(-\omega_k)$ on $q_{\text{anc}}$, where:
     $$\omega_k = 2\pi \sum_{j=1}^{m-k} \frac{c_{k+j}}{2^{j+1}}$$
     conditioned on previously measured classical bits $\{c_{k+1}, \dots, c_m\}$.
  4. Apply Hadamard $H$ on $q_{\text{anc}}$ and measure $q_{\text{anc}} \to c_k$.
  5. Actively reset $q_{\text{anc}} \leftarrow |0\rangle$ using `circuit.reset(q_anc)`.
  6. Repeat for next bit $k-1$.

### C. The Three Experimental Arms for Rigorous Comparison
To prove whether our stack actually solves the problem and outperforms previous work, we execute three distinct arms on real hardware:
1. **Arm 1 (Standard Static NISQ):** Variational / static unmitigated circuit without dynamical decoupling or readout mitigation.
2. **Arm 2 (Raw Dynamic IQPE):** IQPE circuit with mid-circuit measurement and feedforward, but **no** dynamical decoupling during feedforward delays.
3. **Arm 3 (Our Adaptive Mitigated IQPE):** IQPE circuit with:
   - Synchronized **XY4 Dynamical Decoupling** on the system qubit during the $\sim 800\,\text{ns}$ ancilla measurement/reset window.
   - Readout error mitigation on ancilla classical registers.
   - Active leakage tracking.

### D. Quantitative Metrics
- **Chemical Accuracy:** Error $\epsilon = |E_{\text{QPU}} - E_{\text{exact}}|$. (Target: Approaching chemical accuracy $1.6 \times 10^{-3}\,\text{Hartree} \approx 1\,\text{kcal/mol}$).
- **Energy Variance Across Bond Sweep:** Mean absolute error across bond distances $R \in [0.5, 0.74, 1.0, 1.5, 2.0]\,\text{Å}$.

---

## 3. Part 2: Deterministic Quantum Teleportation via Real-Time Feedforward

### A. Physical Problem Formulation
Transfer an unknown quantum state $|\psi\rangle = \cos(\theta/2)|0\rangle + e^{i\phi}\sin(\theta/2)|1\rangle$ from source qubit $q_0$ to destination qubit $q_2$ via transit qubit $q_1$ on the heavy-hex lattice.

### B. Algorithm Architecture
1. **State Preparation:** Prepare test states $\{|0\rangle, |1\rangle, |+\rangle, |-\rangle, |+i\rangle, |-i\rangle\}$ on $q_0$ using parameterized $R_y(\theta), R_z(\phi)$.
2. **Entanglement Distribution:** Create Bell pair between $q_1$ and $q_2$:
   $$|\Phi^+\rangle_{12} = \text{CNOT}(q_1 \to q_2)(H \otimes I)|00\rangle_{12}$$
3. **Bell-State Measurement (BSM):**
   - Apply $\text{CNOT}(q_0 \to q_1)$.
   - Apply $H(q_0)$.
   - Mid-circuit measurement: $c_0 \leftarrow \text{Measure}(q_0)$, $c_1 \leftarrow \text{Measure}(q_1)$.
4. **On-Chip FPGA Feedforward (Dynamic Circuit):**
   - In standard static quantum mechanics:
     $$|\Psi\rangle_{012} = \frac{1}{2} \left[ |00\rangle |\psi\rangle_{2} + |01\rangle (X|\psi\rangle_2) + |10\rangle (Z|\psi\rangle_2) + |11\rangle (XZ|\psi\rangle_2) \right]$$
   - Real-time conditional corrections on $q_2$:
     ```python
     with circuit.if_test((c1, 1)):
         circuit.x(q2)
     with circuit.if_test((c0, 1)):
         circuit.z(q2)
     ```
5. **State Verification (Quantum State Tomography):** Measure $q_2$ in $X, Y, Z$ bases to reconstruct the full density matrix $\rho_{\text{teleported}}$.

### C. The Three Experimental Arms for Rigorous Comparison
1. **Arm 1 (Post-Selected Teleportation - Static Baseline):**
   - Discards all shots where $(c_0, c_1) \neq (0, 0)$.
   - **Flaw:** Only $25\%$ effective throughput; non-deterministic.
2. **Arm 2 (SWAP Network - Unitary Baseline):**
   - Teleportation via 3 physical SWAP gates ($9$ two-qubit gates).
   - **Flaw:** High depth and cumulative 2-qubit gate error ($\sim 15-20\%$).
3. **Arm 3 (Our Adaptive Dynamic Teleportation):**
   - Deterministic feedforward ($100\%$ throughput) + **XY4 pulse train on $q_2$** while $q_0, q_1$ are measured and classical logic resolves in the control rack.

### D. Quantitative Metrics
- **Teleportation State Fidelity:** $\mathcal{F} = \langle \psi | \rho_{q_2} | \psi \rangle$.
- **Classical Communication Threshold:** Prove $\mathcal{F} > \frac{2}{3} \approx 66.7\%$ (the rigorous classical limit of state transfer without quantum entanglement).
- **Deterministic Yield:** Ratio of valid output shots ($\approx 100\%$ for Dynamic vs. $25\%$ for Post-Selected).

---

## 4. End-to-End Execution Plan & Milestones

### Milestone 1: Mathematical Engine & Circuit Generators
- Create `src/adaptive_qec/experiments/molecular_iqpe.py`:
  - Analytical STO-3G molecular integral solver for $H_2$.
  - Dynamic IQPE circuit generator supporting arbitrary bond lengths $R$, bit precision $m$, mid-circuit reset, conditional phase gates, and XY4 decoupling insertion.
- Create `src/adaptive_qec/experiments/deterministic_teleportation.py`:
  - Parameterized quantum state generator for 6 cardinal states.
  - Dynamic teleportation circuit with mid-circuit Bell measurement and conditional $X/Z$ corrections.
  - Tomography basis rotation circuits ($X, Y, Z$ projection).

### Milestone 2: Unit Testing & Algorithmic Validation
- Create `tests/test_molecular_iqpe.py` and `tests/test_teleportation.py`.
- Verify exact state evolution, classical feedforward logic, and integration with Qiskit dynamic circuit syntax.
- Ensure 100% test pass rate across the full repository.

### Milestone 3: Live Hardware Execution on IBM Heron (`ibm_marrakesh`)
- Execute real physical jobs:
  - Part 1: IQPE dissociation curve at equilibrium ($R = 0.7414\,\text{Å}$) and stretched bonds ($R = 1.2\,\text{Å}, 1.8\,\text{Å}$) comparing Unmitigated vs Adaptive DD.
  - Part 2: Deterministic Teleportation of $|+\rangle$ and $|+i\rangle$ states comparing Post-Selected vs Dynamic Feedforward vs SWAP baseline.
- Capture immutable IBM Quantum Runtime Job IDs and full JSON results.

### Milestone 4: Scientific Comparative Analysis & Audit Report
- Compute chemical accuracy $\Delta E$, quantum state fidelity $\mathcal{F}$, and statistical significance ($p$-values, error bars).
- Document honest comparison against prior art (Nature 2023 utility paper, PRX Quantum 2024 dynamic circuit benchmarks).
- Commit code, hardware outputs, and push to GitHub.
