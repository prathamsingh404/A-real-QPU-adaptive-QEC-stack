"""
Unit tests for Deterministic Quantum Teleportation via Real-Time Dynamic Feedforward.
"""

from __future__ import annotations

import math
import pytest

from adaptive_qec.experiments.deterministic_teleportation import (
    DeterministicTeleportation,
    get_cardinal_state,
    CLASSICAL_TELEPORTATION_BOUND,
)


class TestTeleportation:
    """Test suite for deterministic quantum teleportation."""

    def test_get_cardinal_states(self) -> None:
        """Verify cardinal state definitions and Bloch vectors."""
        s0 = get_cardinal_state("|0>")
        assert s0.ideal_bloch_vector == (0.0, 0.0, 1.0)
        s1 = get_cardinal_state("|1>")
        assert s1.ideal_bloch_vector == (0.0, 0.0, -1.0)
        sp = get_cardinal_state("|+>")
        assert sp.ideal_bloch_vector == (1.0, 0.0, 0.0)
        spi = get_cardinal_state("|+i>")
        assert spi.ideal_bloch_vector == (0.0, 1.0, 0.0)
