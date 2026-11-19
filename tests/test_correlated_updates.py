import numpy as np
from adaptive_qec.decoders.correlated_updates import CorrelatedPrior
from adaptive_qec.decoders.fault_library import FaultLibrary
from adaptive_qec.experiments.causal_reweighting import circuit


def test_prior_roundtrip_keeps_correlated_predictions_and_expected_marginals():
    mem = circuit(3,"Z","stationary",0,0.004)
    prior = CorrelatedPrior(mem)
    x = mem.compile_detector_sampler(seed=102030).sample(10000)
    np.testing.assert_array_equal(prior.graph().decode_batch(x,enable_correlations=True),
                                  prior.graph(prior.attenuation).decode_batch(x,enable_correlations=True))
    assert np.max(abs(x.mean(axis=0)-prior.rate(prior.attenuation))) < 0.015


def test_cached_template_identifies_known_qubit_without_observable_labels():
    mem = circuit(3,"Z","stationary",0,0.004)
    lib = FaultLibrary(mem)
    q = lib.targets[4]
    x = circuit(3,"Z","hotspot",0.02,0.004,q).compile_detector_sampler(seed=899).sample(4000)
    assert lib.targets[lib.select(x)] == q
