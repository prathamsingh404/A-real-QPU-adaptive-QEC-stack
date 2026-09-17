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
