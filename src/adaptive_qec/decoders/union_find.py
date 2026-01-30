"""
Distance-weighted cluster growth decoder with Union-Find disjoint-set forest.

Implements a distance-weighted cluster growth and greedy matching decoder
inspired by the cluster-merging principles of topological decoders:
    Delfosse & Nickerson, "Almost-linear time decoding of topological codes"
    Quantum 5, 595 (2021). arXiv:2104.09539

Algorithmic architecture:
    Rather than uniform discrete bucket-growth on the full decoding lattice,
    this implementation uses on-demand Dijkstra shortest-path queries between
    active defect targets, constructs candidate merge events, and performs
    greedy Union-Find merging with lazy observable XOR tracking.
    Complexity per shot is O(k · |E| log |V|) where k is the number of defect
    vertices, |E| is the number of error hyperedges, and |V| is the number
    of detector vertices.

Empirical Performance Characteristics & Scientific Scope:
    - Compared to exact minimum-weight perfect matching (PyMatching v2),
      this heuristic trades matching optimality for fast greedy cluster resolution.
    - Empirical benchmarking indicates a significant accuracy penalty
      (often 25% to >100% higher logical error rate depending on code distance,
      lattice geometry, and noise structure).
    - Consequently, for the confirmatory scientific conclusions of the adaptive
      QEC stack, PyMatching MWPM is utilized as the primary baseline across all
      decoder arms (static, naive reweighted, and gated) to prevent greedy
      heuristic suboptimality from confounding the evaluation of online adaptation.

Algorithm:
    1. Build a detector graph from the Stim DetectorErrorModel
    2. For each syndrome shot:
       a. Identify defect vertices (detectors that fired)
       b. Compute on-demand shortest paths to neighboring defects and boundary
       c. Sort candidate merge events by distance/weight
       d. Merge clusters via union-find, tracking observable XOR along
          merge paths using lazy accumulation with path compression
       e. When odd-parity clusters merge (or connect to boundary),
          resolve parity and accumulate predicted observable flips
"""

from __future__ import annotations

import heapq
import logging
import time
import tracemalloc
from collections import defaultdict
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
        - find(x): root + obs_to_root, amortized O(alpha(N))
        - union(x, y, edge_obs): amortized O(alpha(N))
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

        # Compute obs from ry to rx via the path: ry<-y--edge--x->rx
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


@dataclass
class DetectorGraph:
    """Graph structure from a Stim DetectorErrorModel."""
    num_detectors: int
    num_observables: int
    edges: list[DetectorEdge]
    boundary_node: int
    adjacency: dict[int, list[int]] = field(default_factory=dict)

    # Sparse adjacency for on-demand BFS: node -> [(neighbor, weight, obs_mask)]
    sparse_adj: dict[int, list[tuple[int, float, int]]] = field(default_factory=dict)

    def build_adjacency(self) -> None:
        self.adjacency = {}
        self.sparse_adj = defaultdict(list)
        for i, edge in enumerate(self.edges):
            self.adjacency.setdefault(edge.u, []).append(i)
            self.adjacency.setdefault(edge.v, []).append(i)
            self.sparse_adj[edge.u].append((edge.v, edge.weight, edge.observables))
            self.sparse_adj[edge.v].append((edge.u, edge.weight, edge.observables))


def build_detector_graph(dem: stim.DetectorErrorModel) -> DetectorGraph:
    """Build a detector graph from a Stim DetectorErrorModel.

    Properly decomposes DEM error instructions containing separators (^),
    combines independent error mechanisms on parallel edges, and selects
    the maximum-likelihood observable mask for each edge.

    WARNING -- Hyperedge chain decomposition:
        When a DEM error instruction contains >2 detectors (a hyperedge),
        this function decomposes it into a chain of pairwise edges with the
        same probability. This is an approximation that can change the
        probability distribution of the resulting decoder graph. For most
        surface code circuits with decompose_errors=True, Stim already
        produces pairwise edges, so this fallback rarely triggers. When it
        does, a warning is logged.
    """
    num_detectors = dem.num_detectors
    num_observables = dem.num_observables
    boundary_node = num_detectors

    # Map (u, v) -> dict[obs_mask, probability] where u <= v
    edge_dict: dict[tuple[int, int], dict[int, float]] = {}
    hyperedge_count = 0
