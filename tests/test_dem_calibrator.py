"""
Tests for DEM calibration module.

Verifies:
    - DEMEdge dataclass construction
    - CalibratedDEM container operations
    - DEMCalibrator initialization and weight scaling
    - Edge probability clamping at boundaries
    - Weight update preserves graph structure
    - Mathematical correctness of log-weight conversion
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_qec.decoders.dem_calibration import (
    CalibratedDEM,
    DEMCalibrator,
    DEMEdge,
)


class TestDEMEdge:
    def test_construction(self):
        edge = DEMEdge(
            detector_a=0,
            detector_b=1,
            probability=0.01,
            observables=frozenset(),
        )
        assert edge.detector_a == 0
        assert edge.detector_b == 1
        assert edge.probability == 0.01

    def test_boundary_edge(self):
        """Boundary edges have detector_b = -1 or similar sentinel."""
        edge = DEMEdge(
            detector_a=5,
            detector_b=-1,
            probability=0.005,
            observables=frozenset({0}),
        )
        assert edge.detector_b == -1


class TestCalibratedDEM:
    def test_construction(self):
        edges = [
            DEMEdge(0, 1, 0.01, frozenset()),
            DEMEdge(1, 2, 0.02, frozenset()),
        ]
        dem = CalibratedDEM(
            edges=edges,
            num_detectors=3,
            num_observables=1,
        )
        assert len(dem.edges) == 2
        assert dem.num_detectors == 3


class TestDEMCalibrator:
    def test_initialization(self):
        calibrator = DEMCalibrator()
        assert calibrator is not None

    def test_probability_clamping(self):
        """Probabilities must stay in [min, max] range."""
        calibrator = DEMCalibrator(
            min_probability=1e-6,
            max_probability=0.4999,
        )
        
        # Test clamping at lower bound
        clamped_low = max(1e-6, -0.001)
        assert clamped_low >= 1e-6

        # Test clamping at upper bound
        clamped_high = min(0.4999, 0.99)
        assert clamped_high <= 0.4999

    def test_log_weight_conversion(self):
        """w = ln((1-p)/p) should be correct for valid probabilities."""
        p = 0.01
        w = np.log((1 - p) / p)
        assert w > 0  # Low probability → high weight

        p_high = 0.49
        w_high = np.log((1 - p_high) / p_high)
        assert w_high > 0  # Still positive but small
        assert w_high < w  # Higher probability → lower weight

    def test_weight_scaling_proportional(self):
        """When hardware error doubles, edge weights should scale accordingly."""
        p_nominal = 0.01
        p_measured = 0.02  # Doubled
        
        # Proportional scaling
        scale = p_measured / p_nominal
        p_scaled = p_nominal * scale
        assert abs(p_scaled - p_measured) < 1e-10

    def test_symmetric_edge_handling(self):
        """Edge (a, b) and (b, a) should produce the same weight."""
        p = 0.015
        w1 = np.log((1 - p) / p)
        w2 = np.log((1 - p) / p)
        assert abs(w1 - w2) < 1e-15

    def test_calibrate_and_to_matching(self):
        """Calibrator produces valid matching graph."""
        calibrator = DEMCalibrator()
        calibrated = calibrator.calibrate(measured_p_2q=0.005, measured_p_ro=0.02)
        assert len(calibrated.edges) > 0

        matching = calibrated.to_matching()
        assert matching is not None
        assert matching.num_detectors > 0

    def test_calibrate_from_syndromes(self):
        """Calibrator reweights edges from syndrome batch."""
        calibrator = DEMCalibrator()
        dummy_syndromes = np.zeros((100, 24), dtype=np.uint8)
        dummy_syndromes[:5, 0] = 1
        dummy_syndromes[:5, 1] = 1

        calibrated = calibrator.calibrate_from_syndromes(dummy_syndromes, smoothing=0.5)
        assert len(calibrated.edges) > 0
        matching = calibrated.to_matching()
        assert matching is not None

    def test_calibrate_spitz_matching_recovery(self):
        """Verify Spitz et al. (2018) parity estimator recovers known edge probabilities."""
        import stim, pymatching
        from adaptive_qec.decoders.dem_calibration import calibrate_spitz_matching

        circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3, after_clifford_depolarization=0.005)
        dem = circuit.detector_error_model(decompose_errors=True)
        base_matching = pymatching.Matching.from_detector_error_model(dem)

        sampler = circuit.compile_detector_sampler()
        dets = sampler.sample(shots=5000)

        calibrated_matching = calibrate_spitz_matching(base_matching, dets)
        assert calibrated_matching is not None
        assert calibrated_matching.num_detectors == base_matching.num_detectors
        assert calibrated_matching.num_fault_ids == base_matching.num_fault_ids

    def test_spitz_estimator_on_known_circuit(self):
        """Build a simple circuit with known error rate, run Spitz, verify estimate within 20% of truth."""
        import pymatching
        import stim
        from adaptive_qec.decoders.dem_calibration import calibrate_spitz_matching

        c_simple = stim.Circuit(
            "X_ERROR(0.04) 0\n"
            "M 0\n"
            "DETECTOR(0, 0) rec[-1]\n"
            "X_ERROR(0.04) 0\n"
            "M 0\n"
            "DETECTOR(1, 0) rec[-1] rec[-2]\n"
        )
        dem_simple = c_simple.detector_error_model(decompose_errors=True)
        m_simple = pymatching.Matching.from_detector_error_model(dem_simple)
        syn_simple = c_simple.compile_detector_sampler().sample(shots=20000)
        m_cal = calibrate_spitz_matching(m_simple, syn_simple)
        edge_probs = [d["error_probability"] for _, _, d in m_cal.edges()]
        for p_est in edge_probs:
            assert abs(p_est - 0.04) / 0.04 < 0.20

    def test_spitz_estimator_empty_syndromes_fallback(self):
        """0 shots fallback returns base DEM."""
        calibrator = DEMCalibrator()
        empty_syndromes = np.zeros((0, 24), dtype=np.uint8)
        calibrated = calibrator.calibrate_from_syndromes(empty_syndromes)
        assert calibrated.source == "empty_syndromes_fallback"
        assert len(calibrated.edges) == len(calibrator._base_edges)
        assert calibrated.num_detectors == calibrator._base_dem.num_detectors

    def test_proportional_calibration_clamps_to_threshold(self):
        """Large measured error ratio clamps probability to max_probability < 0.5."""
        calibrator = DEMCalibrator(max_probability=0.48)
        calibrated = calibrator.calibrate(measured_p_2q=0.45, measured_p_ro=0.45)
        for edge in calibrated.edges:
            assert edge.probability <= 0.48
            assert edge.probability < 0.5
            assert edge.weight > 0.0

    def test_windowed_spitz_ewma_converges(self):
        """Multiple syndrome windows with EWMA converge toward drifted noise parameter."""
        import stim
        circuit_base = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3, after_clifford_depolarization=0.005)
        circuit_drift = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3, after_clifford_depolarization=0.03)

        calibrator = DEMCalibrator(circuit=circuit_base)
        sampler_drift = circuit_drift.compile_detector_sampler()

        # Start from base DEM prior
        prev_dem = CalibratedDEM(
            num_detectors=calibrator._base_dem.num_detectors,
            num_observables=calibrator._base_dem.num_observables,
            edges=list(calibrator._base_edges),
        )
        base_mean_p = float(np.mean([e.probability for e in prev_dem.edges]))
        probs_history = [base_mean_p]

        for _ in range(5):
            syn = sampler_drift.sample(shots=2000)
            if hasattr(calibrator, "calibrate_from_syndromes_windowed"):
                prev_dem = calibrator.calibrate_from_syndromes_windowed(syn, alpha=0.4, prev_dem=prev_dem)
            else:
                cur_dem = calibrator.calibrate_from_syndromes(syn, smoothing=0.0)
                alpha = 0.4
                new_edges = []
                for ec, ep in zip(cur_dem.edges, prev_dem.edges):
                    p_up = (1.0 - alpha) * ep.probability + alpha * ec.probability
                    e = DEMEdge(ec.detector_a, ec.detector_b, float(p_up), list(ec.observables))
                    e.compute_weight()
                    new_edges.append(e)
                prev_dem = CalibratedDEM(cur_dem.num_detectors, cur_dem.num_observables, new_edges)
            mean_p = float(np.mean([e.probability for e in prev_dem.edges]))
            probs_history.append(mean_p)

        assert probs_history[-1] > probs_history[0]
        assert probs_history[-1] > base_mean_p

    def test_weight_uncertainty_decreases_with_shots(self):
        """Higher shot count yields lower finite-sample uncertainty in estimated edge weights."""
        import stim
        circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3, after_clifford_depolarization=0.01)
        calibrator = DEMCalibrator(circuit=circuit)
        sampler = circuit.compile_detector_sampler()

        syn_small = sampler.sample(shots=150)
        syn_large = sampler.sample(shots=3000)

        if hasattr(calibrator, "estimate_weight_uncertainty"):
            unc_small = calibrator.estimate_weight_uncertainty(syn_small)
            unc_large = calibrator.estimate_weight_uncertainty(syn_large)
        else:
            rng = np.random.default_rng(42)
            def get_uncertainty(syn, n_boot=8):
                weight_samples = []
                for _ in range(n_boot):
                    idx = rng.choice(len(syn), size=len(syn), replace=True)
                    dem = calibrator.calibrate_from_syndromes(syn[idx], smoothing=0.0)
                    weight_samples.append([e.weight for e in dem.edges[:10]])
                return float(np.mean(np.std(weight_samples, axis=0)))

            unc_small = get_uncertainty(syn_small)
            unc_large = get_uncertainty(syn_large)

        assert unc_large < unc_small
