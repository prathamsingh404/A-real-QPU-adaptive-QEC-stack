"""
Dynamical Decoupling (DD) integration for idle-time noise suppression.

When a qubit is idle during a QEC round (e.g. waiting for a two-qubit gate
on neighbors to finish, or during SWAP routing on heavy-hex), it accumulates
coherent ZZ crosstalk and non-Markovian low-frequency dephasing.

Indiscriminate DD can hurt: each DD pulse incurs gate error. An optimal
framework applies DD selectively — only when the dephasing noise prevented
exceeds the pulse error penalty.

Supported sequences:
    - CPMG: X - X (order 2, basic dephasing echo)
    - XY4: X - Y - X - Y (order 4, robust against pulse rotation errors)
    - XY8: XY4 + rotated XY4 (order 8, high fidelity)

Sources:
    - IBM Qiskit "Orbit" dynamical decoupling framework
