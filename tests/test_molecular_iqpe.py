"""
Unit tests for Molecular Quantum Chemistry via Iterative Quantum Phase Estimation (IQPE).
"""

from __future__ import annotations

import math
import pytest

from adaptive_qec.experiments.molecular_iqpe import (
    MolecularIQPE,
    get_h2_sto3g_coefficients,
    CHEMICAL_ACCURACY_HARTREE,
)


class TestMolecularIQPE:
    """Test suite for molecular IQPE calculations and circuits."""

    def test_h2_sto3g_coefficients_equilibrium(self) -> None:
        """Verify H2 Hamiltonian coefficients at equilibrium bond length (0.7414 Angstroms)."""
        coeffs = get_h2_sto3g_coefficients(0.7414)
        assert coeffs.r_angstrom == 0.7414
        assert coeffs.nuclear_repulsion > 0.7  # 1 / 1.4 bohr ~ 0.714
        assert math.isclose(coeffs.g0, -1.4229, rel_tol=1e-3)
        assert math.isclose(coeffs.g1, 0.3879, rel_tol=1e-3)
        assert math.isclose(coeffs.g2, 0.1812, rel_tol=1e-3)
        # Ground state energy around -1.137 Hartrees
        assert -1.2 < coeffs.exact_ground_energy < -1.1

    def test_h2_coefficients_dissociation_limit(self) -> None:
        """Verify coefficients scale physically as bond is stretched."""
        r_eq = get_h2_sto3g_coefficients(0.7414)
        r_stretched = get_h2_sto3g_coefficients(2.0)
        # Exchange coupling g2 should decay as bond stretches
        assert r_stretched.g2 < r_eq.g2
        # Nuclear repulsion decays with 1/R
        assert r_stretched.nuclear_repulsion < r_eq.nuclear_repulsion

    def test_build_circuit_without_dd(self) -> None:
        """Test building IQPE circuit without dynamical decoupling."""
        iqpe = MolecularIQPE(r_angstrom=0.7414, num_bits=3, tau=1.0, apply_dd=False)
        qc = iqpe.build_circuit()
        assert qc.num_qubits == 2
        assert qc.num_clbits == 3
        # Should have mid-circuit measurements and resets
        op_names = [inst.operation.name for inst in qc.data]
        assert "measure" in op_names
        assert "reset" in op_names
        assert "crz" in op_names or "rz" in op_names

    def test_build_circuit_with_dd(self) -> None:
        """Test building IQPE circuit with XY4 dynamical decoupling."""
        iqpe = MolecularIQPE(r_angstrom=0.7414, num_bits=3, tau=1.0, apply_dd=True)
        qc = iqpe.build_circuit()
        op_names = [inst.operation.name for inst in qc.data]
        # XY4 decoupling injects X and Y gates on system qubit
        assert "x" in op_names
        assert "y" in op_names

    def test_decode_energy_from_bitstrings(self) -> None:
        """Test decoding physical bitstrings into ground state energy."""
        iqpe = MolecularIQPE(r_angstrom=0.7414, num_bits=3, tau=1.0)
        counts = {"000": 800, "001": 150, "111": 50}
        results = iqpe.decode_energy_from_bitstrings(counts)
        assert results["r_angstrom"] == 0.7414
        assert results["most_likely_bitstring"] == "000"
        assert results["total_shots"] == 1000
        assert "qpu_ground_energy_hartree" in results
        assert "exact_ground_energy_hartree" in results
        assert "error_hartree" in results

    def test_empty_counts_raises(self) -> None:
        """Test that empty counts dictionary raises ValueError."""
        iqpe = MolecularIQPE(r_angstrom=0.7414)
        with pytest.raises(ValueError, match="Empty counts"):
            iqpe.decode_energy_from_bitstrings({})
