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
