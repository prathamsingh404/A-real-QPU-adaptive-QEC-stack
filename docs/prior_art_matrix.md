# Comparative Prior Art Matrix & Literature Grounding

**Manuscript Reference Document (TASK-103)**  
**Associated Target Publication**: *IEEE Transactions on Quantum Engineering* / *Quantum Science and Technology*  
**Paper Title**: *"When Should Quantum Error Correction Adapt? Confidence- and Cost-Constrained Decoder Selection under Non-Stationary Noise"*

---

## 1. Prior Art Comparison Matrix

| Approach / Work | Primary Target | Adaptation Mechanism | Considers Switching / Latency Cost? | Has Explicit Abstention (`STAY`)? | Observation Boundary Enforced? | Regret / Theoretical Bounds? | Real QPU Validation |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Static MWPM**<br>(Dennis et al. 2002; Fowler 2012) | Stationary Pauli noise | None (fixed prior) | N/A | Always stays | Yes (syndromes only) | Static threshold proofs | Extensive across all platforms |
| **Online Reweighted MWPM**<br>(Bhardwaj et al., *PRX Quantum* 2026) | Continuous drift in syndrome rates | Sliding-window detector error model reweighting | ❌ No (continuous reweighting) | ❌ No (always updates weights) | Yes (syndromes only) | None reported | Simulated drift |
| **AlphaQubit**<br>(DeepMind / Google, *Nature* 2024) | High-accuracy decoding on 72Q Sycamore | Deep recurrent transformer neural network | ❌ No (evaluates every round) | ❌ No (single fixed neural policy) | Yes (soft/analog readouts + syndromes) | Empirical neural scaling | 72-qubit Sycamore QPU |
| **GreenPeas (JIT QEC)**<br>(Chao et al., 2024) | Runtime compilation overhead | Just-in-time compilation of Pauli frames and graphs | ⚠️ Partial (compilation latency) | ❌ No (compiles when requested) | Classical compiler layer | None | Emulated runtime |
| **RL QEC Control**<br>(Google Quantum AI, *Nature* 2026) | Low-level physical pulse / calibration tuning | Continuous deep reinforcement learning | ⚠️ Partial (loss function reward) | ❌ No (continuous policy gradient) | Yes (calibration + leak telemetry) | Empirical RL convergence | Willow QPU |
| **Leakage-Aware (LCD)**<br>(Brown 2020; Suchara 2015) | Physical leakage out of computational subspace | Post-selection or static graph modification | ❌ No | ❌ No | Requires dedicated leak flags | Heuristic threshold | Simulated / Trapped Ion |
| **CG-CCADS (This Work)** | **Meta-level decoder selection under non-stationary noise** | **3-Gate Confidence & Cost Constrained Selection** | **✅ Explicitly quantified** ($C_{\text{switch}} + C_{\text{latency}} + C_{\text{compute}}$) | **✅ Core primitive** (Default is zero-cost `STAY`) | **✅ Strict `ObservationFrame`** (Zero oracle leakage) | **✅ Sub-Gaussian false switch & piecewise regret bounds** | **Interleaved A/B runs on IBM Heron** |

---

## 2. Complementary Positioning vs. Nature 2024 / 2026 Literature

### 2.1 AlphaQubit (Google DeepMind, *Nature* 2024)
- **What AlphaQubit Achieves**: Breakthrough neural decoder accuracy for a single fixed-code instance using multi-layer transformers and soft analog readout values.
- **The Open Gap**: AlphaQubit does not address *meta-level controller decisions*: when should a quantum system incur the substantial computational overhead of running an inference model versus remaining with ultra-fast lookup or MWPM?
- **Our Positioning**: CG-CCADS sits *above* specialized decoders (including AlphaQubit, Reweighted MWPM, and Union-Find). When AlphaQubit is registered as candidate rung $B_4$ in our `BaselineLadder`, CG-CCADS decides whether the noise drift justifies the neural inference latency tax.

### 2.2 Google Willow RL Control (*Nature* 2026)
- **What Google Willow Achieves**: Physical-layer pulse shaping, frequency tuning, and calibration optimization to reduce raw physical error rates.
- **Our Positioning**: Physical calibration and classical decoder selection operate on orthogonal planes:
  - Google Willow operates at the **physical Hamiltonian level** (controlling microwave pulses and gate parameters).
  - CG-CCADS operates at the **classical algorithmic level** (selecting the optimal decoding graph and inference algorithm under observed syndrome statistics).
  - Crucially, this work proves that physical interventions (e.g., dynamic decoupling pulses) carry an overhead tax that can degrade logical performance if applied indiscriminately without a decision gate.

---

## 3. Four Core Contributions of This Manuscript (TASK-107)

1. **Discovery & Formalization of the Adaptation Paradox in QEC**:
   We prove analytically and demonstrate empirically that naive continuous adaptation degrades logical fidelity (e.g., $0.170320 \to 0.217560$, a 27.74% loss). We identify the four physical adaptation taxes that dictate when adaptation is mathematically unprofitable.
2. **The CG-CCADS Algorithmic Architecture**:
   A decision-theoretic meta-controller featuring three sequential gates:
   - *Gate 1: Multi-Timescale Drift Detection* ($\Delta r > \epsilon_{\text{drift}}$)
   - *Gate 2: Counterfactual Disagreement Gating* ($D(d_{\text{cand}}, d_{\text{active}}) > \delta_{\text{min}}$)
   - *Gate 3: Cost-Constrained Lower Confidence Bound (LCB)* ($\text{LCB}(\Delta) > C_{\text{adapt}}$)
   equipped with dual-threshold hysteresis, dead zone, probationary trials, and automatic rollback.
3. **Rigorous Analytical Bounds & Operational Phase Diagram**:
   Derivations of:
   - Analytical upper bound on false-switching probability under sub-Gaussian noise: $P(\text{False Switch}) \le \exp\left(-\frac{N \tau^2}{2 \sigma^2}\right)$.
   - Minimum drift boundary threshold $A_{\text{drift}}^*$.
   - Cumulative regret bound under piecewise-stationary drift: $\mathcal{O}(S \sqrt{T} + S \cdot C_{\text{switch}})$.
   - Complete 2D empirical phase diagram establishing the boundaries between `STAY`, `ADAPT`, and the `THRASHING_ZONE`.
4. **Independent Trajectory Statistical Protocol & IBM Heron Hardware Validation**:
   Establishment of independent noise trajectories as the primary statistical unit (30 held-out test seeds) with pre-registered confirmatory endpoints ($\bar{\Delta}_{\text{LER}}$), paired trajectory analysis, and verified interleaved $A/B/A/B$ execution on real IBM Heron superconducting QPUs.
