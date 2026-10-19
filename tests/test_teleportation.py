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

    def test_compute_fidelity_ideal_plus_state(self) -> None:
        """Verify fidelity calculation for ideal |+> state counts."""
        state = get_cardinal_state("|+>")
        dt = DeterministicTeleportation(target_state=state, regime="dynamic")
        # In X-basis for |+>, outcome 0 should dominate
        counts_x = {"000": 950, "100": 50}
        # In Y-basis for |+>, outcomes 0 and 1 are 50/50
        counts_y = {"000": 500, "100": 500}
        # In Z-basis for |+>, outcomes 0 and 1 are 50/50
        counts_z = {"000": 500, "100": 500}

        results = dt.compute_fidelity_from_tomography(counts_x, counts_y, counts_z)
        assert results["state_fidelity"] > 0.90
        assert results["surpasses_classical_limit"] is True
        assert results["deterministic_yield_rate"] == 1.0

    def test_post_selected_yield_rate(self) -> None:
        """Verify post-selected regime discards non-(0,0) BSM shots."""
        state = get_cardinal_state("|0>")
        dt = DeterministicTeleportation(target_state=state, regime="post_selected")
        # 4 BSM branches: 00, 01, 10, 11 (each ~25% frequency)
        counts_z = {"000": 250, "001": 250, "010": 250, "011": 250}
        counts_x = {"000": 250, "001": 250, "010": 250, "011": 250}
        counts_y = {"000": 250, "001": 250, "010": 250, "011": 250}

        results = dt.compute_fidelity_from_tomography(counts_x, counts_y, counts_z)
        # Only the '000' branch is retained out of the 1000 total per basis
        assert results["deterministic_yield_rate"] < 0.35
