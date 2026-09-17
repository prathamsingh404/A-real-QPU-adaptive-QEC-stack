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

