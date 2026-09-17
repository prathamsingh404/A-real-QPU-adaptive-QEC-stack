"""
Surface code embedding on hardware topologies.

Maps logical surface code qubits (data + ancilla) onto physical hardware
qubits, accounting for connectivity constraints.

The key challenge: surface codes require a square grid, but IBM's heavy-hex
lattice has max degree 3 and irregular connectivity. Embedding requires
SWAP routing, which introduces idle time and noise overhead.

Embedding quality metrics:
