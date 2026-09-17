"""
Surface code embedding on hardware topologies.

Maps logical surface code qubits (data + ancilla) onto physical hardware
qubits, accounting for connectivity constraints.

The key challenge: surface codes require a square grid, but IBM's heavy-hex
lattice has max degree 3 and irregular connectivity. Embedding requires
SWAP routing, which introduces idle time and noise overhead.

Embedding quality metrics:
    - SWAP count: total SWAPs needed per QEC round
    - Circuit depth overhead: increase in circuit depth vs ideal
    - Idle time: total idle slots where qubits accumulate noise
    - Connectivity deficit: fraction of required edges missing

Sources:
    - IBM/ETH Zurich "fold-unfold" embedding strategy
    - Lao & Almudever, "Mapping of lattice surgery-based quantum circuits
      on surface code architectures" (2019)
"""

