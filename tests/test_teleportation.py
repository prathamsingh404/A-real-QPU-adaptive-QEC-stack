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

    def test_unknown_cardinal_state_raises(self) -> None:
        """Verify invalid state name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown cardinal state"):
            get_cardinal_state("nonexistent")

    def test_build_dynamic_circuit(self) -> None:
        """Verify dynamic circuit construction with mid-circuit BSM and feedforward."""
        state = get_cardinal_state("|+>")
        dt = DeterministicTeleportation(target_state=state, regime="dynamic", apply_dd=True)
        qc = dt.build_circuit(measurement_basis="Z")
        assert qc.num_qubits == 3
        assert qc.num_clbits == 3  # 2 BSM bits + 1 output bit
        op_names = [inst.operation.name for inst in qc.data]
        assert "measure" in op_names
        assert "cx" in op_names
        assert "h" in op_names

    def test_build_swap_network_circuit(self) -> None:
        """Verify static SWAP network baseline circuit."""
        state = get_cardinal_state("|+>")
        dt = DeterministicTeleportation(target_state=state, regime="swap_network", apply_dd=False)
        qc = dt.build_circuit(measurement_basis="X")
        op_names = [inst.operation.name for inst in qc.data]
        assert "swap" in op_names
