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
