# Executive Summary

We propose **five** complementary research contributions to transform the AdaptiveQEC framework from a simulation prototype into a real-world, hardware-validated adaptive QEC system. Each contribution is grounded in recent literature but pushes beyond existing work. In particular:

1. **Online Adaptive Strategy Selection via Multi-Armed Bandits.** Replace static cost-minimization with an online learning controller that *estimates and updates the expected logical error and latency* of each QEC strategy (decoder+mitigation action) under the current hardware conditions. By treating each strategy as an “arm” and using a contextual bandit or UCB algorithm, the controller **learns from actual outcomes** to balance exploration and exploitation. This is novel (compared to fixed analytical cost or black-box RL) and aligns with industry needs for interpretable, low-latency control.  

2. **Adaptive X/Z Stabilizer Scheduling.** Dynamically adjust the *ratio and ordering of X-basis vs Z-basis stabilizer measurement rounds* based on observed error trends. For example, if relaxation (T1) is dominating (bit-flips), run extra X-basis rounds; if dephasing (T2) dominates, run extra Z-basis rounds. This extends heavy-hex code flexibility and leverages control in Qiskit Runtime. To our knowledge, adaptively varying X/Z cycle balance in response to real-time syndrome data is new.  

3. **Data-Driven Decoder Calibration.** Continuously calibrate decoder parameters from real hardware data. For example, refine error model probabilities or edge weights (including correlated/hyperedge errors) using syndrome statistics or hardware calibrations (T1, T2, gate/measurement error rates) gathered during idle/ancilla circuits. Unlike fixed error models, this “online calibration” makes the decoder **hardware-aware** and can feed more accurate matching (e.g. via PyMatching’s correlated matching).  

4. **Qiskit Runtime Integration and Experimentation.** Implement the adaptive loop inside IBM’s Qiskit Runtime environment with mid-circuit measurement capabilities. Though not novel academically, doing so is critical for industry relevance. We will engineer an end-to-end pipeline that issues shots, gathers syndromes, runs the adaptive controller logic (bandit or decision rules), and issues updated circuits – all within the latency and budget of the IBM cloud system.  

5. **Statistical Decision Framework for Adaptation.** Formalize the switching logic with statistical hypothesis tests. For example, use a two-proportion Z-test or Wilson confidence intervals to verify that one strategy’s estimated logical error is significantly lower than another’s before switching (this complements hysteresis). This makes the controller’s decisions **statistically sound** and mitigates overfitting to random fluctuations.  

Each proposal (1–5) is implementable within ~3–6 months and can be validated first by simulation and then on IBM heavy-hex hardware (Marrakesh/Heron). We summarize the **novelty vs prior work** for each, required resources, experimental plans, data, metrics, failure modes, and risk mitigation below. The top two ideas (Bandit Control and Adaptive Scheduling) are detailed further with algorithms, analysis, and experimental protocols. Recommended primary sources and images/diagrams are provided for clarity.

**Key Sources to Explore First:** For background and tools, see the PyMatching library documentation and Stim documentation (for syndrome simulation), the IBM heavy-hex lattice description, Google’s recent RL-based QEC (Nature 2026), the Local Clustering decoder paper (Nat. Commun. 2025–26), and the new NASA paper on heavy-hex QEC. Also review the latest IBM Qiskit Runtime documentation for dynamic circuits.

**Implementation Roadmap:** An integrated roadmap with milestones and person-weeks is given at the end.  

---

## 1. Online Adaptive Strategy via Multi-Armed Bandit

