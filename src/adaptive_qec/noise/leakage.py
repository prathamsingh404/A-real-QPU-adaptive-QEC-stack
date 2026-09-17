"""
Leakage detection from QEC syndrome data.

Leakage — a qubit transitioning from the computational subspace {|0⟩, |1⟩}
to higher energy states {|2⟩, |3⟩, ...} — is a persistent, correlated error
that standard QEC can't handle. A leaked qubit produces incorrect syndrome
information for *every subsequent round* until it's detected and reset.

Detection approach:
    A leaked qubit produces a *persistent* defect pattern: the same detector
    fires every round (or nearly every round). This is distinct from a
