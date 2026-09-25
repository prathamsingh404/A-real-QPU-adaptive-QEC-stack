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

**Description (what, why).** Replace the current fixed cost function approach with an *online learning controller*. Treat each combined decoder+mitigation action (e.g. “MWPM+no-DD”, “UF+XY8”, etc.) as an arm of a multi-armed bandit. As syndrome data arrives, estimate the **empirical logical error rate** (and decode latency) of each arm in the current noise context, then choose arms to minimize long-run error and overhead. This allows the controller to **learn from actual outcomes** rather than relying on assumed models. It naturally handles non-stationary noise by adapting estimates over time (e.g. via discounted averaging). The bandit approach (e.g. UCB, Thompson sampling) is interpretable and avoids complex neural networks, aligning with industry trends for explainable QEC control.

**Novelty vs Prior Work.** Prior works used *offline* or fixed controllers. Google’s Nature (2026) experiment did RL on fixed drift models, but used deep learning controllers to optimize continuous control parameters. In contrast, our proposal uses a simple **contextual bandit** strategy that only learns discrete action-values. To our knowledge, bandit-based adaptive QEC with explicit confidence bounds is not reported. Nickerson & Brown (Q-2019) adapted decoding heuristically, and Nvidia’s (unpublished) AI pre-decoders address unknown noise offline, but neither performed online arm-selection with statistical guarantees. This approach is thus both novel and feasible. 

**Hardware/Software Resources.** 
- **Hardware:** Any IBM heavy-hex QPU (e.g. 65–127 qubit Heron) or simulator for initial tests.  
- **Software:** Qiskit for circuit execution, Stim+PyMatching (and possibly sinter) for simulation. Python libraries for bandit algorithms (or simple custom code). Access to real-time (or batched) syndrome data via Qiskit Runtime or iterative programs. Classical computing (laptop/cluster) for simulation of bandit learning (negligible cost).

**Expected Experiments (Simulation → Real).** 

- *Simulation Stage:* Use the existing AdaptiveQEC simulator stack (Stim+PyMatching) to create non-stationary noise scenarios (e.g. drift in T1/T2 or bursts). Simulate repeated QEC cycles with different actions, and implement a bandit policy (e.g. UCB1, ε-greedy with decaying ε, or contextual bandit if using observed syndrome stats as context). Evaluate how quickly the bandit identifies the best action for each regime. Compare to static or hysteresis controllers. 

- *Hardware Stage:* Implement a proof-of-concept on IBM hardware. For example, run alternating batches of QEC circuits under controlled noise conditions (IBM hardware drifts naturally; we might induce artificial asymmetry by idle delays or by choosing qubits with different T1). Use a Qiskit Runtime program that performs rounds, sends syndromes to a simple controller (bandit logic in Python), and then reconfigures upcoming circuits. Collect enough shots to update action estimates (e.g. 10k shots per setting). 

**Data to Collect (Provenance):** 
- For each shot: the full syndrome (“dets”) for each round, chosen action, and whether a logical failure occurred. Record exact timestamps or round indices.  
- Hardware calibration data at time of experiment: T1_i, T2_i, single- and two-qubit gate errors, readout error (for each physical qubit) from IBM’s daily calibration. Label these as “calibration metadata” (provenance: measured, simulation vs experiment).  
- Controller state: counts of successes/failures per action, action choices over time.  
- Additional: if possible, record raw readout bitstrings to independently recompute logical outcomes.  

**Evaluation Metrics / Statistical Tests:**  
- **Primary metric:** Estimated Logical Error Rate (LER) for each strategy (and overall). Compute Wilson 95% confidence intervals for each LER. Use two-proportion Z-tests or overlap of confidence intervals to test if the chosen adaptive strategy significantly outperforms baselines (e.g. best static strategy) at p<0.05.  
- **Latency / Overhead:** Measure decoding time (from post-processing) and any added runtime overhead from bandit computation. Ensure all strategies meet real-time constraints (PyMatching decoding time is ~0.1–1 μs/round).  
- **Regret:** In simulation, track cumulative regret or performance loss of the bandit vs omniscient policy.  
