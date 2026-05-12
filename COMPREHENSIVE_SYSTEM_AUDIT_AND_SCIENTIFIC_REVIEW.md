# Comprehensive System Audit, Forensic Execution Analysis & Scientific Review
## Real-QPU Adaptive Quantum Error Correction Stack on Heavy-Hex Architectures

> **Document Type**: SYSTEM AUDIT, CODEBASE VERIFICATION & PEER-REVIEW STRESS-TEST  
> **Date**: September 26, 2026  
> **Target Processor Target**: IBM Heron r2 (156-qubit Heavy-Hex, `ibm_marrakesh`, `ibm_kingston`) via Qiskit Runtime  
> **Core Repository**: `src/adaptive_qec/`, `tests/`, `configs/`, `experiments/`  
> **Execution Status**: Complete end-to-end execution of all tests, experiments, pipelines, and modules completed. All errors, anti-patterns, simulations, and hardcoded logic cataloged without modifying source code.

---

## Executive Summary & Uncompromising Verdict

A comprehensive, end-to-end execution and forensic inspection of every module, pipeline, test suite, and experiment script in this repository was conducted. While the architectural ambition of building an adaptive, hardware-aware QEC stack for IBM heavy-hex superconducting QPUs is theoretically compelling, the current implementation contains **critical execution errors**, **severe interface mismatches**, **simulated shortcuts masquerading as real experiments**, and **claims that diverge from physical hardware realities**.

### Key Findings at a Glance:
1. **Test Suite Status**: **42 Failures and Errors** (25 failed, 17 errors, 195 passed out of 237 collected items in `pytest`).
2. **Experiment Pipeline Status**: **4 out of 5 experiment runners crash immediately upon launch** due to import errors (`ImportError: cannot import name 'wilson_score_ci'`, `cannot import name 'StatisticalTestResult'`). The single experiment that runs (`adaptive_vs_static.py`) yields an empirical improvement of $+1.02\%$ with $p = 0.801$ (statistically **insignificant** at $\alpha = 0.05$).
3. **Forensic Discovery of "Phantom" Implementations**:
   - In `adaptive_scheduling.py`, the code claims to compare `BALANCED`, `X_HEAVY`, `Z_HEAVY`, and `ADAPTIVE` schedules, but `_build_biased_circuit()` builds the **exact same Stim circuit** for all four arms. The schedule parameter is completely ignored in circuit construction. Furthermore, $X$ vs $Z$ defects are counted by naively bisecting the detector array (`:n//2` vs `n//2:`), which does not correspond to physical check operators.
   - In `bandit_vs_static.py`, line 280 contains the comment `# Decode using MWPM (both arms use same decoder for now)` and executes `mwpm_matcher.decode_batch()` for every controller arm (including `static_uf_xy4` and bandit-selected Union-Find). The decoder chosen by the bandit is never actually executed!
   - In `full_adaptive.py`, lines 325–326 literally copy the adaptive arm's LER into the static baseline arrays: `self._static_mwpm_lers.append(ler) # Same decoder, so same LER`.
   - In `qiskit_loop.py`, the "dry-run" hardware mode does not execute Stim circuits; it draws independent Bernoulli coin flips `rng.random() < p_phys`, generating syndrome data completely devoid of topological graph structure.
4. **Physical Reality Mismatch**: Claims of "real-time adaptive QEC" running in Python across Qiskit Runtime cloud sessions conflate **intra-circuit feedforward** ($<1\ \mu\text{s}$ required; IBM hardware does $\sim 600\ \text{ns}$ on-chip) with **inter-batch cloud session communication** ($2\text{s}$ to $300\text{s}$ latency over HTTP).

This audit documents every issue with exact line numbers, stack traces, mathematical analysis, and a five-reviewer expert debate, followed by rigorous proposed solutions.

---

## 1. Complete Test Suite & Execution Diagnostics

Execution command: `.venv/Scripts/pytest -v --tb=short`  
Results: **195 PASSED, 25 FAILED, 17 ERRORS (Total: 237 tests in 6.87s)**

```
=========================== short test summary info ===========================
FAILED tests/test_bandit_controller.py::TestBanditArm::test_arm_label - TypeError: BanditArm.__init__() got an unexpected keyword argument 'dd_sequence'
FAILED tests/test_bandit_controller.py::TestBanditArm::test_build_arm_set_default - AssertionError: assert 4 == 6
FAILED tests/test_bandit_controller.py::TestExp3Controller::test_initialization - TypeError: Exp3Controller.__init__() got an unexpected keyword argument 'arms'
FAILED tests/test_bandit_controller.py::TestExp3Controller::test_probability_distribution_sums_to_one - TypeError: Exp3Controller.__init__() got an unexpected keyword argument 'arms'
FAILED tests/test_bandit_controller.py::TestExp3PController::test_initialization - TypeError: Exp3PController.__init__() got an unexpected keyword argument 'arms'
FAILED tests/test_bandit_controller.py::TestDASEController::test_initialization - TypeError: DASEController.__init__() got an unexpected keyword argument 'arms'
FAILED tests/test_dem_calibrator.py::TestDEMCalibrator::test_initialization - TypeError: DEMCalibrator.__init__() missing 1 required positional argument: 'circuit'
FAILED tests/test_dem_calibrator.py::TestDEMCalibrator::test_probability_clamping - TypeError: DEMCalibrator.__init__() got an unexpected keyword argument 'min_probability'
FAILED tests/test_scenarios.py::TestNoiseSnapshot::test_to_calibration_dict - TypeError: NoiseSnapshot.__init__() got an unexpected keyword argument 'gate_error_1q'
FAILED tests/test_scenarios.py::TestScenarioConfig::test_invalid_steps_raises - Failed: DID NOT RAISE ValueError
FAILED tests/test_scenarios.py::TestStationaryScenario::test_constant_error_rate - AttributeError: 'NoiseSnapshot' object has no attribute 'gate_error_2q'
FAILED tests/test_scenarios.py::TestStationaryScenario::test_returns_valid_snapshots - AttributeError: 'NoiseSnapshot' object has no attribute 'gate_error_2q'
FAILED tests/test_scenarios.py::TestDriftScenario::test_monotonic_drift - AttributeError: 'NoiseSnapshot' object has no attribute 'gate_error_2q'
FAILED tests/test_scenarios.py::TestDriftScenario::test_t1_degrades - AttributeError: 'NoiseSnapshot' object has no attribute 't1_us'
FAILED tests/test_scenarios.py::TestBurstScenario::test_produces_spikes - TypeError: burst_scenario() got an unexpected keyword argument 'burst_probability'
FAILED tests/test_scenarios.py::TestMultiPhaseScenario::test_has_transitions - TypeError: multi_phase_scenario() got an unexpected keyword argument 'total_steps'
FAILED tests/test_scenarios.py::TestConvenienceConstructors::test_multi_phase_constructor - TypeError: multi_phase_scenario() got an unexpected keyword argument 'total_steps'
FAILED tests/test_sprt.py::TestSPRTState::test_initial_state - TypeError: SPRTState.__init__() missing 1 required positional argument: 'challenger_label'
FAILED tests/test_sprt.py::TestSPRTState::test_reset - TypeError: SPRTState.__init__() missing 1 required positional argument: 'challenger_label'
FAILED tests/test_sprt.py::TestSPRTEngine::test_initialization - TypeError: SPRTEngine.__init__() got an unexpected keyword argument 'p0'
FAILED tests/test_sprt.py::TestSPRTEngine::test_boundaries_correct - TypeError: SPRTEngine.__init__() got an unexpected keyword argument 'p0'
FAILED tests/test_sprt.py::TestSPRTEngine::test_update_with_successes - TypeError: SPRTEngine.__init__() got an unexpected keyword argument 'p0'
FAILED tests/test_sprt.py::TestSPRTEngine::test_update_with_failures - TypeError: SPRTEngine.__init__() got an unexpected keyword argument 'p0'
FAILED tests/test_sprt.py::TestSPRTEngine::test_decision_under_strong_evidence - TypeError: SPRTEngine.__init__() got an unexpected keyword argument 'p0'
FAILED tests/test_sprt.py::TestSPRTController::test_initialization - TypeError: SPRTController.__init__() got an unexpected keyword argument 'arms'
ERROR tests/test_bandit_controller.py::TestExp3Controller::test_decide_returns_valid_action - TypeError: HardwareState.__init__() got an unexpected keyword argument 'error_rate'
ERROR tests/test_bandit_controller.py::TestExp3Controller::test_update_shifts_weights - TypeError: HardwareState.__init__() got an unexpected keyword argument 'error_rate'
ERROR tests/test_bandit_controller.py::TestExp3Controller::test_numerical_stability_under_extreme_rewards - TypeError: HardwareState.__init__() got an unexpected keyword argument 'error_rate'
ERROR tests/test_bandit_controller.py::TestExp3Controller::test_reset_restores_uniform - TypeError: HardwareState.__init__() got an unexpected keyword argument 'error_rate'
ERROR tests/test_bandit_controller.py::TestExp3PController::test_decide_and_update - TypeError: HardwareState.__init__() got an unexpected keyword argument 'error_rate'
ERROR tests/test_bandit_controller.py::TestExp3PController::test_probabilities_valid - TypeError: HardwareState.__init__() got an unexpected keyword argument 'error_rate'
ERROR tests/test_bandit_controller.py::TestDASEController::test_forced_exploration - TypeError: HardwareState.__init__() got an unexpected keyword argument 'error_rate'
ERROR tests/test_bandit_controller.py::TestDASEController::test_convergence_to_best_arm - TypeError: HardwareState.__init__() got an unexpected keyword argument 'error_rate'
ERROR tests/test_bandit_controller.py::TestDASEController::test_drift_detection_reactivates_arms - TypeError: HardwareState.__init__() got an unexpected keyword argument 'error_rate'
ERROR tests/test_bandit_controller.py::TestDASEController::test_summary_contains_expected_keys - TypeError: HardwareState.__init__() got an unexpected keyword argument 'error_rate'
ERROR tests/test_bandit_controller.py::TestBanditIntegration::test_full_loop_exp3 - TypeError: HardwareState.__init__() got an unexpected keyword argument 'error_rate'
ERROR tests/test_bandit_controller.py::TestBanditIntegration::test_full_loop_with_reset - TypeError: HardwareState.__init__() got an unexpected keyword argument 'error_rate'
ERROR tests/test_sprt.py::TestSPRTController::test_decide_returns_action - TypeError: HardwareState.__init__() got an unexpected keyword argument 'error_rate'
ERROR tests/test_sprt.py::TestSPRTController::test_update_with_reward - TypeError: HardwareState.__init__() got an unexpected keyword argument 'error_rate'
ERROR tests/test_sprt.py::TestSPRTController::test_full_loop - TypeError: HardwareState.__init__() got an unexpected keyword argument 'error_rate'
ERROR tests/test_sprt.py::TestSPRTController::test_reset - TypeError: HardwareState.__init__() got an unexpected keyword argument 'error_rate'
ERROR tests/test_sprt.py::TestSPRTController::test_summary - TypeError: HardwareState.__init__() got an unexpected keyword argument 'error_rate'
================== 25 failed, 195 passed, 17 errors in 6.87s ==================
```

### Detailed Taxonomy of Test Failures

#### Bug Category 1: `HardwareState` Signature Mismatch Across Test Fixtures
- **Files Affected**: [tests/conftest.py](file:///d:/Antigravity%20IDE/A%20real-QPU%20adaptive%20QEC%20stack/tests/conftest.py#L26-L65), causing 17 test errors in `test_bandit_controller.py` and `test_sprt.py`.
- **Error Type**: `TypeError: HardwareState.__init__() got an unexpected keyword argument 'error_rate'`
- **Root Cause**: In [src/adaptive_qec/controller/controller.py](file:///d:/Antigravity%20IDE/A%20real-QPU%20adaptive%20QEC%20stack/src/adaptive_qec/controller/controller.py#L58-L84), `HardwareState` is defined as:
  ```python
  @dataclass(frozen=True)
  class HardwareState:
      defect_rate: float
      drift_magnitude: float
      drift_status: DriftStatus
      burst_active: bool
      leakage_fraction: float
      t1_mean_us: float = 0.0
      t2_mean_us: float = 0.0
      p_1q: float = 0.0
      p_2q: float = 0.0
      p_ro: float = 0.0
  ```
  However, `tests/conftest.py` instantiates it using non-existent attributes:
  ```python
  HardwareState(
      error_rate=0.005,
      t1_us=200.0,
      t2_us=150.0,
      readout_error=0.01,
      gate_error_1q=0.0005,
      gate_error_2q=0.003,
  )
  ```
- **Impact**: Any test relying on the `dummy_hardware_state`, `dephasing_dominated_state`, or `relaxation_dominated_state` fixtures immediately raises an unhandled exception before reaching the controller execution logic.

#### Bug Category 2: Bandit Controller Arm Initialization Disconnect
- **Files Affected**: [src/adaptive_qec/controller/bandit.py](file:///d:/Antigravity%20IDE/A%20real-QPU%20adaptive%20QEC%20stack/src/adaptive_qec/controller/bandit.py#L84-L120), [tests/test_bandit_controller.py](file:///d:/Antigravity%20IDE/A%20real-QPU%20adaptive%20QEC%20stack/tests/test_bandit_controller.py)
- **Error Types**:
  1. `TypeError: BanditArm.__init__() got an unexpected keyword argument 'dd_sequence'`
  2. `TypeError: Exp3Controller.__init__() got an unexpected keyword argument 'arms'`
  3. `AssertionError: assert 4 == 6`
- **Root Cause**:
  - `BanditArm` in `bandit.py` has fields `(decoder, dd_policy, index)`, but test code passes `dd_sequence`.
  - `Exp3Controller`, `Exp3PController`, and `DASEController` hardcode `self._arms = build_arm_set()` inside their `__init__` and take only `gamma` and `weights`, rejecting any custom `arms` parameter passed by the test suite or caller.
  - `build_arm_set()` hardcodes 4 arms: $\{ \text{MWPM}, \text{UF} \} \times \{ \text{NONE}, \text{XY4} \}$. The test suite expects 6 arms (including $\text{XY8}$).

#### Bug Category 3: SPRT Implementation and State Inconsistencies
- **Files Affected**: [src/adaptive_qec/controller/sprt.py](file:///d:/Antigravity%20IDE/A%20real-QPU%20adaptive%20QEC%20stack/src/adaptive_qec/controller/sprt.py#L55-L105), [tests/test_sprt.py](file:///d:/Antigravity%20IDE/A%20real-QPU%20adaptive%20QEC%20stack/tests/test_sprt.py)
- **Error Types**:
  1. `TypeError: SPRTState.__init__() missing 1 required positional argument: 'challenger_label'`
  2. `TypeError: SPRTEngine.__init__() got an unexpected keyword argument 'p0'`
  3. `TypeError: SPRTController.__init__() got an unexpected keyword argument 'arms'`
- **Root Cause**:
  - `SPRTEngine.__init__` was designed to take `(alpha, beta, delta, max_samples)`. However, `tests/test_sprt.py` attempts to pass standard Bernoulli SPRT parameters `(alpha, beta, p0, p1)`.
  - `SPRTState` defined `challenger_label: str` as a non-default positional argument, but test fixtures call `SPRTState()` with default construction.
  - `SPRTController` hardcodes its internal arm list to 4 elements and rejects `arms`.

#### Bug Category 4: Noise Scenario Data Model Semantic Drift
- **Files Affected**: [src/adaptive_qec/engine/scenarios.py](file:///d:/Antigravity%20IDE/A%20real-QPU%20adaptive%20QEC%20stack/src/adaptive_qec/engine/scenarios.py#L44-L65), [tests/test_scenarios.py](file:///d:/Antigravity%20IDE/A%20real-QPU%20adaptive%20QEC%20stack/tests/test_scenarios.py)
- **Error Types**:
  1. `TypeError: NoiseSnapshot.__init__() got an unexpected keyword argument 'gate_error_1q'`
  2. `AttributeError: 'NoiseSnapshot' object has no attribute 'gate_error_2q'`
  3. `AttributeError: 'NoiseSnapshot' object has no attribute 't1_us'`
  4. `TypeError: burst_scenario() got an unexpected keyword argument 'burst_probability'`
  5. `TypeError: multi_phase_scenario() got an unexpected keyword argument 'total_steps'`
- **Root Cause**:
  - `NoiseSnapshot` defines fields: `step, p_1q, p_2q, p_ro, t1_mean_us, t2_mean_us`.
  - `tests/test_scenarios.py` was written assuming an older schema: `gate_error_1q, gate_error_2q, t1_us, t2_us`.
  - `burst_scenario()` accepts `(total_steps, burst_at, duration)`, but tests pass `burst_probability`.
  - `multi_phase_scenario()` accepts `phase_durations: list[int]`, but tests pass `total_steps: int`.

#### Bug Category 5: DEMCalibrator API Mismatch
- **Files Affected**: [src/adaptive_qec/decoders/dem_calibration.py](file:///d:/Antigravity%20IDE/A%20real-QPU%20adaptive%20QEC%20stack/src/adaptive_qec/decoders/dem_calibration.py#L110-L135), [tests/test_dem_calibrator.py](file:///d:/Antigravity%20IDE/A%20real-QPU%20adaptive%20QEC%20stack/tests/test_dem_calibrator.py)
- **Error Types**:
  1. `TypeError: DEMCalibrator.__init__() missing 1 required positional argument: 'circuit'`
  2. `TypeError: DEMCalibrator.__init__() got an unexpected keyword argument 'min_probability'`
- **Root Cause**: `DEMCalibrator` requires a base `stim.Circuit` as its first argument to parse the baseline DEM and nominal noise rates. The test suite attempts to instantiate it in isolation with `DEMCalibrator(min_probability=1e-6)`.

---

## 2. Experiment Pipeline Execution Audit & Traceback Analysis

All 5 experiment scripts located in `src/adaptive_qec/experiments/` were executed directly within the active environment:

### Experiment 1: `adaptive_vs_static.py`
- **Execution**: `.venv/Scripts/python -m adaptive_qec.experiments.adaptive_vs_static`
- **Result**: **SUCCESSFUL EXECUTION (Code 0)**
- **Empirical Output**:
  ```
  ========================================================================
  ADAPTIVE vs STATIC QEC — EXPERIMENT RESULTS
  ========================================================================
    Distance: d=3, Rounds: 3
    Windows: 50 x 200 shots
    Total shots per arm: 10,000

    STATIC MWPM:   LER = 0.107500   95% CI [0.101579, 0.113722]
    STATIC UF+XY4: LER = 0.109000   95% CI [0.103041, 0.115259]
    ADAPTIVE:      LER = 0.106400   95% CI [0.100507, 0.112596]

    Best static arm:  static_mwpm
    Improvement:      +1.02%
    z-statistic:      -0.2517
    p-value:          0.801288
    Significant (5%): False
    Mode switches:    1
  ========================================================================
  ```
- **Scientific Critical Evaluation**:
  - The README and documentation previously claimed a clear, statistically verified advantage for the adaptive controller.
  - In this verified live run, the adaptive arm achieved an LER of $0.1064$ vs Static MWPM at $0.1075$.
  - The difference is **11 errors out of 10,000 shots** ($+1.02\%$ relative improvement).
  - The two-proportion $z$-test yields $p = 0.801288$. The $95\%$ Wilson score confidence intervals completely overlap: $[0.1005, 0.1126]$ vs $[0.1016, 0.1137]$.
  - **Conclusion**: The adaptive controller's advantage under this synthetic drift model is **statistically indistinguishable from random shot-noise fluctuation**.

---

### Experiment 2: `bandit_vs_static.py`
- **Execution**: `.venv/Scripts/python -m adaptive_qec.experiments.bandit_vs_static`
- **Result**: **CRASHED IMMEDIATELY (Code 1)**
- **Traceback**:
  ```python
  Traceback (most recent call last):
    File "<frozen runpy>", line 198, in _run_module_as_main
    File "<frozen runpy>", line 88, in _run_code
    File "D:\Antigravity IDE\A real-QPU adaptive QEC stack\src\adaptive_qec\experiments\bandit_vs_static.py", line 67, in <module>
      from adaptive_qec.analysis.significance import (
  ImportError: cannot import name 'StatisticalTestResult' from 'adaptive_qec.analysis.significance'
  ```
- **Root Cause**: `significance.py` defines `HypothesisTestResult`, but `bandit_vs_static.py` imports non-existent `StatisticalTestResult` and `wilson_score_ci`.

---

### Experiment 3: `adaptive_scheduling.py`
- **Execution**: `.venv/Scripts/python -m adaptive_qec.experiments.adaptive_scheduling`
- **Result**: **CRASHED IMMEDIATELY (Code 1)**
- **Traceback**:
  ```python
  Traceback (most recent call last):
    File "<frozen runpy>", line 198, in _run_module_as_main
    File "<frozen runpy>", line 88, in _run_code
    File "D:\Antigravity IDE\A real-QPU adaptive QEC stack\src\adaptive_qec\experiments\adaptive_scheduling.py", line 49, in <module>
      from adaptive_qec.analysis.significance import wilson_score_ci
  ImportError: cannot import name 'wilson_score_ci' from 'adaptive_qec.analysis.significance'
  ```
- **Root Cause**: Unverified import name. `wilson_score_ci` is implemented in `adaptive_vs_static.py` as `wilson_ci()` and in `statistics.py` as `confidence_interval(..., method="wilson")`, but was never exported from `significance.py`.

---

### Experiment 4: `hardware_baseline.py`
- **Execution**: `.venv/Scripts/python -m adaptive_qec.experiments.hardware_baseline`
- **Result**: **CRASHED DURING ANALYSIS (Code 1)**
- **Traceback**:
  ```python
  2026-09-26 18:04:06,521 [INFO] __main__: Starting hardware baseline: backend=ibm_marrakesh, d=3, dry_run=True
  2026-09-26 18:04:06,522 [INFO] adaptive_qec.runtime.qiskit_loop: DRY RUN mode: using local simulator instead of ibm_marrakesh
  2026-09-26 18:04:06,563 [INFO] adaptive_qec.runtime.qiskit_loop: Runtime loop b3ef2fd4 complete: 20 batches, 20000 total shots, overall LER=0.026900, elapsed=0.0s
  Traceback (most recent call last):
    File "D:\Antigravity IDE\A real-QPU adaptive QEC stack\src\adaptive_qec\experiments\hardware_baseline.py", line 162, in _analyze
      from adaptive_qec.analysis.significance import wilson_score_ci
  ImportError: cannot import name 'wilson_score_ci' from 'adaptive_qec.analysis.significance'
  ```
- **Root Cause**: The runtime loop executed 20 batches of synthetic coin flips in 0.04s, but crashed when attempting to calculate confidence intervals due to the missing import.

---

### Experiment 5: `full_adaptive.py`
- **Execution**: `.venv/Scripts/python -m adaptive_qec.experiments.full_adaptive`
- **Result**: **CRASHED IMMEDIATELY (Code 1)**
- **Traceback**:
  ```python
  Traceback (most recent call last):
    File "D:\Antigravity IDE\A real-QPU adaptive QEC stack\src\adaptive_qec\experiments\full_adaptive.py", line 66, in <module>
      from adaptive_qec.analysis.significance import wilson_score_ci, welch_t_test
  ImportError: cannot import name 'wilson_score_ci' from 'adaptive_qec.analysis.significance'
  ```
- **Root Cause**: Identical import failure of `wilson_score_ci` from `adaptive_qec.analysis.significance`.

---

## 3. Forensic Discovery of Hardcoded, Mocked, Simulated & Phantom Mechanisms

Beyond explicit Python syntax and import errors, an architectural audit of the codebase reveals deep discrepancies between what the documentation claims to accomplish and what the code actually executes:

### Phantom Mechanism 1: Unused Schedules in `adaptive_scheduling.py`
In [src/adaptive_qec/experiments/adaptive_scheduling.py](file:///d:/Antigravity%20IDE/A%20real-QPU%20adaptive%20QEC%20stack/src/adaptive_qec/experiments/adaptive_scheduling.py#L181-L242):
```python
def _run_fixed_schedule(
    self,
    eta: float,
    p_x: float,
    p_z: float,
    schedule_type: ScheduleType,
) -> BiasPointResult:
    """Run experiment with a fixed stabilizer schedule."""
    circuit = self._build_biased_circuit(p_x, p_z)
    sampler = circuit.compile_detector_sampler()
    ...
```
- **The Problem**: Notice that `schedule_type` is accepted as an argument, but **never passed to `_build_biased_circuit()`**!
- Inside `_build_biased_circuit(p_x, p_z)`:
  ```python
  p_total = p_x + p_z
  circuit = stim.Circuit.generated(
      "surface_code:rotated_memory_z",
      distance=d,
      rounds=r,
      after_clifford_depolarization=p_total,
      before_round_data_depolarization=p_total,
      ...
  )
  return circuit
  ```
- **The Reality**: The Stim built-in `surface_code:rotated_memory_z` generates a fixed, symmetric rotated surface code. The `schedule_type` variable is discarded. The code claimed to benchmark `BALANCED` vs `X_HEAVY` vs `Z_HEAVY`, but was actually simulating the **identical standard circuit four times** with different random seeds.

### Phantom Mechanism 2: Arbitrary Detector Bisection as "X vs Z Defects"
In `adaptive_scheduling.py` (lines 277–279) and `full_adaptive.py` (lines 258–260):
```python
n_det = detection_events.shape[1]
x_det = int(np.sum(detection_events[:, : n_det // 2]))
z_det = int(np.sum(detection_events[:, n_det // 2 :]))
```
- **The Problem**: The code assumes the first $N/2$ detectors in the Stim circuit are $X$-stabilizers and the remaining $N/2$ are $Z$-stabilizers.
- **The Physical Reality**: In Stim's rotated surface code generator, detectors are ordered spatially and temporally across rounds. Each round interweaves $X$ and $Z$ checks across the 2D lattice. Bisections across the middle split early rounds from late rounds, or upper spatial coordinates from lower spatial coordinates. The "imbalance ratio" $\Delta_{XZ}$ was calculating spatial/temporal round disparities, not Pauli error bias.

### Phantom Mechanism 3: The Ignored Bandit Action in `bandit_vs_static.py`
In [src/adaptive_qec/experiments/bandit_vs_static.py](file:///d:/Antigravity%20IDE/A%20real-QPU%20adaptive%20QEC%20stack/src/adaptive_qec/experiments/bandit_vs_static.py#L273-L290):
```python
for ctrl_name, ctrl in self._controllers.items():
    ctrl.observe(hw_state)
    action = ctrl.decide()

    # Decode using MWPM (both arms use same decoder for now)
    predictions = mwpm_matcher.decode_batch(detection_events)
    logical_errors = int(np.sum(np.any(pred_flat != obs_flat, axis=1)))
```
- **The Problem**: The bandit controller observes the state and decides an action (e.g. `action.decoder = UNION_FIND`, `action.dd_sequence = XY4`).
- **The Reality**: The decoder that actually decodes the syndromes is hardcoded to `mwpm_matcher` for **all** arms! Even the static baseline `static_uf_xy4` is evaluated using MWPM. Furthermore, no dynamical decoupling is injected into `noisy_circuit`.
- **Consequence**: Every single controller receives identical predictions and identical rewards. The bandit is learning over a dummy environment where actions have zero causal effect on rewards.

### Phantom Mechanism 4: Self-Copying Baselines in `full_adaptive.py`
In [src/adaptive_qec/experiments/full_adaptive.py](file:///d:/Antigravity%20IDE/A%20real-QPU%20adaptive%20QEC%20stack/src/adaptive_qec/experiments/full_adaptive.py#L324-L327):
```python
# Track static baselines (same syndrome data)
self._static_mwpm_lers.append(ler)  # Same decoder, so same LER
self._static_uf_lers.append(ler)
```
- **The Problem**: The experiment claims to demonstrate the superiority of the full adaptive stack over static baselines.
- **The Reality**: The static baseline arrays (`_static_mwpm_lers` and `_static_uf_lers`) simply append the adaptive arm's LER (`ler`) on every window. A comparative plot between "Adaptive" and "Static MWPM" would show identical, superimposed curves.

### Simulated Mechanism 5: Coin-Flip Syndromes in `qiskit_loop.py`
In [src/adaptive_qec/runtime/qiskit_loop.py](file:///d:/Antigravity%20IDE/A%20real-QPU%20adaptive%20QEC%20stack/src/adaptive_qec/runtime/qiskit_loop.py#L531-L550):
```python
def _generate_synthetic_data(self, shots: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    n_detectors = 24
    p_phys = 0.005 + 0.001 * np.sin(2 * np.pi * self.total_batches / 50)
    syndromes = rng.random((shots, n_detectors)) < p_phys
    observables = rng.random((shots, 1)) < (p_phys * 1.5)
    return syndromes.astype(np.uint8), observables.astype(np.uint8)
```
- **The Problem**: In "dry-run" mode, instead of calling Stim to simulate the circuit and extract topological detector samples, it generates syndromes via independent Bernoulli coin flips ($P = p_{\text{phys}}$).
- **The Reality**: Independent Bernoulli bits do not satisfy the stabilizer code graph constraints:
  1. Isolated defects appear without matching partners or boundaries.
  2. The parity of defects does not match topological homology.
  3. When PyMatching or Union-Find decodes independent random bits, it performs massive, meaningless matching across the entire lattice, producing artificial logical failure rates that have no physical connection to QEC codes.

### Flawed Hardware Abstraction in `qiskit_loop.py`
In lines 516–522 of `qiskit_loop.py`:
```python
# Last column(s) are observables, rest are detectors
if n_bits > 1:
    syndromes = raw[:, :-1]
    observables = raw[:, -1:]
