"""
Union-Find decoder for surface codes.

Implements the weighted Union-Find decoder from:
    Delfosse & Nickerson, "Almost-linear time decoding of topological codes"
    Quantum 5, 595 (2021). arXiv:2104.09539

Algorithmic complexity: O(N · α(N)) per shot, where N = number of detectors
and α is the inverse Ackermann function (effectively constant ≤ 4).

This is fundamentally faster than MWPM's O(N³) but typically ~8-15% higher
logical error rate. The tradeoff matters at d ≥ 7 where MWPM becomes
latency-prohibitive for real-time feedback.

Algorithm:
    1. Build a detector graph from the Stim DetectorErrorModel
    2. For each syndrome shot:
       a. Identify defect vertices (detectors that fired)
       b. Grow clusters by processing edges in weight order (cheapest first)
       c. Merge clusters via union-find, tracking observable XOR along
          merge paths using lazy accumulation with path compression
       d. When two odd-parity clusters merge (or odd meets boundary),
          compute the correction from the accumulated observable XOR
    3. Observable tracking uses lazy XOR: each node stores the XOR from
       itself to its parent, and find() compresses while accumulating

Key data structures:
    - Union-Find forest with path compression, union by rank, and
      per-node observable-XOR-to-parent tracking
    - Pre-sorted edge list by weight for growth phase
"""

from __future__ import annotations

import heapq
import logging
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import stim

from adaptive_qec.decoders.base import Correction, Decoder, DecoderMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Union-Find forest with observable tracking
# ---------------------------------------------------------------------------

class UnionFindForest:
    """
    Weighted union-find (disjoint set) with observable XOR tracking.

    Each node stores obs_to_parent: the XOR of observables along the
    path from this node to its parent. Path compression maintains this
    invariant by accumulating XORs during find().

    Supports:
        - find(x): root + obs_to_root, amortized O(α(N))
        - union(x, y, edge_obs): amortized O(α(N))
        - cluster parity and boundary tracking
    """

    def __init__(self, n: int, n_obs: int = 1) -> None:
        self.n = n
        self.n_obs = n_obs
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int32)
        self.size = np.ones(n, dtype=np.int32)
        self.parity = np.zeros(n, dtype=np.int32)
        self.boundary_connected = np.zeros(n, dtype=np.bool_)
        # Observable XOR from node to its parent
        self.obs_to_parent = np.zeros((n, n_obs), dtype=np.uint8)

    def find(self, x: int) -> int:
        """Find root with path compression (recursively accumulates obs_to_parent)."""
        p = self.parent[x]
        if p != x:
            root = self.find(p)
            if p != root:
                self.obs_to_parent[x] ^= self.obs_to_parent[p]
                self.parent[x] = root
            return root
        return x

    def obs_to_root(self, x: int) -> np.ndarray:
        """Get observable XOR from x to its root. Calls find() first."""
        self.find(x)  # Ensure path is compressed
        return self.obs_to_parent[x].copy()

    def union(self, x: int, y: int, edge_obs_mask: int = 0) -> int:
        """
        Union by rank, tracking observable XOR along the merge edge.

        The edge from x to y has observable bitmask edge_obs_mask.
        After union, the obs_to_parent chain correctly tracks the
        cumulative XOR from any node to the new root.

        Returns the new root.
        """
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return rx

        # Compute obs from ry to rx via the path: ry←y—edge—x→rx
        obs_y_to_ry = self.obs_to_parent[y]
        obs_x_to_rx = self.obs_to_parent[x]

        edge_obs = np.zeros(self.n_obs, dtype=np.uint8)
        for i in range(self.n_obs):
            if edge_obs_mask & (1 << i):
                edge_obs[i] = 1

        obs_ry_to_rx = obs_y_to_ry ^ edge_obs ^ obs_x_to_rx

        # Union by rank
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx

        self.parent[ry] = rx
        self.obs_to_parent[ry] = obs_ry_to_rx.copy()
        self.size[rx] += self.size[ry]
        self.parity[rx] += self.parity[ry]
        self.boundary_connected[rx] |= self.boundary_connected[ry]

        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

        return rx

    def is_even(self, x: int) -> bool:
        """Check if the cluster containing x has even parity."""
        root = self.find(x)
        return (self.parity[root] % 2 == 0) or bool(self.boundary_connected[root])


# ---------------------------------------------------------------------------
# Graph representation
# ---------------------------------------------------------------------------

@dataclass
class DetectorEdge:
    """An edge in the detector graph."""
    u: int
    v: int
    weight: float
    observables: int  # bitmask
    is_boundary: bool = False

