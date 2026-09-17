"""
Leakage detection from QEC syndrome data.

Leakage — a qubit transitioning from the computational subspace {|0⟩, |1⟩}
to higher energy states {|2⟩, |3⟩, ...} — is a persistent, correlated error
that standard QEC can't handle. A leaked qubit produces incorrect syndrome
information for *every subsequent round* until it's detected and reset.

Detection approach:
    A leaked qubit produces a *persistent* defect pattern: the same detector
    fires every round (or nearly every round). This is distinct from a
    transient error, which produces defects in isolated rounds.

    We detect leakage by computing the temporal autocorrelation of each
    detector's firing history. High autocorrelation at lag 1 indicates
    a persistent defect — the signature of leakage.

Sources:
    - Google AlphaQubit — leakage handling via neural network decoder
    - IBM Heron r2 — native leakage reduction circuits
    - Battistel et al., "Hardware-efficient leakage-reduction scheme for
      quantum error correction with superconducting transmon qubits" (2021)
