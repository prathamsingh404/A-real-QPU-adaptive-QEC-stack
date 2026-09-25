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
- **Adaptation Speed:** Time or number of shots until the bandit identifies the current best arm. This can be quantified by how quickly logical failure rates converge.  
- For all metrics, use at least ~10k shots per point to keep statistical uncertainty low (Wilson CI width ≈1% for LER~10%).  

**Failure Modes and Risk Mitigation:** 
- *Too slow adaptation:* If the hardware noise changes faster than data can be collected, the bandit may chase noise. Mitigation: use discounting (EWMA) in reward estimates or combine with drift detection to “reset” learning on regime change.  
- *Insufficient exploration:* If one arm looks best early, bandit might exploit prematurely. Use UCB or forced exploration to ensure long-term optimality.  
- *Hardware limitations:* Qiskit cloud queue times might hinder on-the-fly adaptation. Mitigation: use Qiskit Runtime or submit consecutive small jobs simulating adaptation (simulate control offline first).  
- *Interference effects:* Frequent switching of circuits may confuse calibration. Mitigation: ensure enough shots per action for stable estimates; use bootstrapping.  

**Novelty vs Prior Work (with citations):** This approach explicitly leverages **online statistical learning** to adapt QEC strategy. While adaptive decoding is known (e.g. Nickerson & Brown, Quantum J. 2019 and Google’s RL QEC (Nature 2026)), those works did *offline* adaptation or used deep RL. Our bandit-based scheme is simpler, interpretable, and novel in the QEC context. It directly addresses *industry concerns* (exploit known decoders, avoid black-box NN) and fits within short development cycles.

## 2. Adaptive X/Z Stabilizer Scheduling

**Description.** Implement a controller that **dynamically varies the frequency of X-basis vs Z-basis stabilizer checks** based on the current noise profile. For example, if syndrome analysis or hardware calibration indicates strong relaxation (short T1 causing bit-flips), perform extra X-stabilizer rounds (which detect Z errors) or vice versa. Concretely, one can design circuits where some rounds omit either the X- or Z-checks (simulating biased repetition codes) and switch between these modes. The controller monitors syndrome imbalance (e.g. higher rate of one type of detection event) and accordingly adjusts the schedule.

**Novelty vs Prior Work.** Fixed QEC schedules (alternating X- and Z-check rounds) are the norm. Some works studied *basis-biased codes* (e.g. the XZZX code) for static noise bias, but *real-time adaptive scheduling* has not been demonstrated. This idea is new: using syndrome feedback to choose the next circuit variant. It leverages IBM’s ability to customize circuits per batch. It also aligns with “hardware-aware” design by exploiting known asymmetry (heavy-hex qubits often have shorter T1 than T2). 

**Hardware/Software Resources.** Same QPUs and Qiskit as above. Additional resource: the ability to compile and run different stabilizer circuits (with different orders of X/Z checks) quickly. Stim and PyMatching can simulate biased schedules. 

**Expected Experiments (Simulation → Real).** 

- *Simulation:* Build circuit variants with, e.g., 2X+1Z, 1X+2Z patterns, etc. Simulate under noise models where $p_X\neq p_Z$ (e.g. $T_1<T_2$). Verify which schedule yields lower logical error (and latency). Then implement an adaptive rule: if recent rounds show more Z-detections, increase X-check rounds. Compare adaptive schedule vs best fixed schedule. 

- *Hardware:* Choose a heavy-hex subset (distance-3 or 5) and run QEC rounds with alternating schedules. Use qubits with known T1/T2 imbalance. Collect syndrome for equal numbers of X- and Z-heavy cycles to verify difference. Then implement the rule (e.g., two experiments: one with static 1:1 schedule, one with adaptive 2:1 when needed) and compare LER. 

**Data to Collect:** Syndromes and logical outcomes for each circuit type. Label each shot with the schedule type. Also log hardware parameters (T1, T2) from calibrations. It is crucial to track which rounds were X-heavy vs Z-heavy so we can correlate syndrome patterns to schedule performance. 

**Evaluation Metrics / Statistical Tests:** Compare logical error rates between static vs adaptive scheduling. Use paired tests if comparing the same noise conditions. A Chi-squared or two-proportion Z-test can assess if differences in failure rates are significant. We also measure cycle latency (different sequences may have different gate counts). A possible metric is “logical error per unit time” to capture both error suppression and speed. 

**Failure Modes and Mitigation:** 
- *Noisy syndrome detection:* If syndrome statistics are too noisy to detect bias, schedule changes may misfire. Mitigate by using EWMA smoothing or requiring persistent trend before switching (hysteresis).  
- *Overhead costs:* Changing schedule adds overhead (compiling new circuits, possible idle times). Ensure the potential error suppression justifies extra complexity. If not, fall back to static schedule.  
- *Hardware calibration drift:* Frequent changes in circuit may be sensitive to drift. Mitigate by testing schedule choices under stable conditions first.  

**Novelty:** This contribution goes beyond existing QEC practice by making the **code itself adaptive**. It leverages hardware-specific error biases (heavy-hex qubits and cross-resonance gates cause known asymmetries) in real-time. To our knowledge, no prior work has implemented live switching of check schedules based on feedback. It is implementable via IBM’s programmable circuits and addresses industry need for hardware-optimized QEC.

## 3. Data-Driven Decoder Calibration

**Description.** Continuously refine decoder parameters from live data. For instance, use the observed syndrome frequencies or calibration metadata to update the **error probabilities** assigned to edges in the Detector Error Model (DEM). Or use measured syndrome correlations to infer correlated/hyperedge errors and enable **correlated matching** in PyMatching. In practice, this could mean periodically running short calibration circuits (e.g. prepare states or run stabilizers without correction) to gather error statistics, then updating the decoder’s weight tables. 

**Novelty vs Prior Work.** Standard practice assumes static noise models or uses manufacturer datasheets. Recent works (Lee et al. 2026) improved performance by “detailed noise characterization” and measurement soft info, but they did this as an offline pre-step. We propose *online and automated* calibration during operation, without human intervention. This dynamic adjustment is novel and makes decoding truly hardware-aware. It also ties into upcoming correlated matching features of PyMatching. 

**Resources.** Same hardware. Software: Stim/PyMatching pipelines already support customized DEMs (see [29] above). Will need small calibration routines (e.g. idle or single gates) and scripts to fit error rates from measurement outcomes. 

**Experiments.** 
- *Simulation:* Inject a known bias or correlation into error model (e.g. more Z-errors on certain qubit). Run the default decoder vs a decoder with weights tuned to the model. Confirm that adaptive calibration (e.g. inferring increased $p_Z$ or linking correlated events) improves logical fidelity. 
- *Hardware:* Run short experiments (like repeated single-qubit gates or readouts) to measure actual error rates. Update the decoder’s input DEM (PyMatching weighted graph) with these values. Then run full QEC and compare to using default/assumed error model. 

**Data Collected:** 
- Raw measurement outcomes from calibration circuits (counts for |0>/<1> to estimate readout error, parity checks to estimate gate error asymmetry). 
- Syndrome from QEC runs for analysis. 
- Metadata: which DEM parameters were used in each run. 

**Metrics:** 
- Improvement in logical error rate due to calibration. 
- We can use likelihood-based metrics: e.g. log-likelihood of observed syndromes under the old vs new model. 
- Use AIC/BIC or simple histogram matching to show decoder is better fit to data.  

**Failure/Risks:** 
- *Noisy calibrations:* If calibration data is too short, weight estimates will have large error. Mitigate by running enough shots or using Bayesian priors. 
- *Miscalibration:* Over-fitting to transient noise could hurt decode. Mitigate by smoothing (EWMA) the inferred parameters over time. 
- *Complexity:* Updating the DEM (possibly correlated edges) must be fast enough for the controller. PyMatching supports quick reweighting (a few ms), so this is low risk. 

This contribution bridges simulation and hardware by ensuring the decoder “knows” the real device. It complements contributions (1)–(2) by sharpening the error models used in the adaptive controller.

## 4. Qiskit Runtime Closed-Loop Implementation

**Description.** Develop a Qiskit Runtime program (or iterative loop) that implements the adaptive control loop on actual hardware. This involves writing circuits for each candidate strategy, dispatching shots, retrieving syndromes, running the controller logic (bandit or scheduling), and using Qiskit’s parameterized jobs to switch strategies between runs. 

**Novelty vs Prior Work.** While Qiskit’s runtime supports dynamic circuits, to our knowledge no one has demonstrated a full adaptive QEC loop on IBM Q hardware (the recent Google experiment used their own control hardware). This contribution is largely engineering-focused, but it is critical to transition from simulation to a real demonstration. It also yields an “artifact” (runtime code) that is exactly the kind of industry-relevant deliverable sought. 

**Resources.** IBM Qiskit Runtime access with ample credits. Qiskit Pulse (if needed for tailored pulses, though not mandatory). The existing AdaptiveQEC code and Stim can be ported or interfaced with Qiskit jobs. 

