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
