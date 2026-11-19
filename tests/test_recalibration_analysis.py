"""Boundary cases for inference and the exploratory decoded-edge occurrence baseline."""
import numpy as np
import pymatching
import pytest

from adaptive_qec.analysis.causal_results import holm, signed_rank_p
from adaptive_qec.decoders.graph_fit import SpitzGraph
from scripts.replay_alignment import trace


def test_signed_rank_exact_with_ties_and_zero():
    assert signed_rank_p([0,0]) == 1
    assert signed_rank_p([0,1,1,2]) == pytest.approx(0.25)
    assert signed_rank_p([1,-1]) == 1
    assert holm({"a":0.01,"b":0.02,"c":0.99}) == pytest.approx({"a":0.03,"b":0.04,"c":0.99})


def test_alignment_counts_boundary_occurrences():
    graph = pymatching.Matching()
    graph.add_boundary_edge(0, fault_ids={0}, weight=1, error_probability=0.2)
    shots = np.array([[0],[1],[0],[0],[0]], dtype=np.uint8)
    fit = SpitzGraph(graph)
    np.testing.assert_allclose(trace(graph,fit,shots), [0.2])
    assert fit.matching(trace(graph,fit,shots),0).decode_batch(shots).tolist() == shots.tolist()
