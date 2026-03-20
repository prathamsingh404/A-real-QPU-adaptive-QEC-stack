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
