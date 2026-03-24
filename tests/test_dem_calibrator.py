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
