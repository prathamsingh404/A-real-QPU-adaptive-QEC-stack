"""
Mathematical correctness tests for the Spitz et al. (2018) parity-covariance estimator.

Verifies:
1. Zero covariance yields zero edge error rate (uncorrelated detectors).
2. Maximal covariance yields error rate approaching 0.5 (perfectly correlated detectors).
3. Parity-covariance formula matches exact closed-form analytical solution.
4. Parity product identity for boundary estimator: 1 - 2*mean = (1 - 2*pb) * prod(1 - 2*pe).
5. Finite-sample variance inflates substantially at small window sizes (W=50 vs W=2000).
"""

from __future__ import annotations

import numpy as np
import pymatching
import pytest
import stim

from adaptive_qec.decoders.dem_calibration import DEMCalibrator, calibrate_spitz_matching


def _spitz_closed_form(mean_u: float, mean_v: float, mean_uv: float) -> float:
    """Analytical Spitz parity-covariance estimator for 2 detectors."""
    num = 4.0 * (mean_uv - mean_u * mean_v)
    den = 1.0 - 2.0 * mean_u - 2.0 * mean_v + 4.0 * mean_uv
    if abs(den) < 1e-12:
        return 0.0
    val = 1.0 - num / den
    val = max(val, 0.0)
    return float(0.5 - 0.5 * np.sqrt(val))


def test_spitz_zero_covariance_gives_base_rate():
    """Uncorrelated detectors (zero covariance) yield estimated edge probability p ~ 0."""
    p_u = 0.05
    p_v = 0.05
    # Zero covariance means E[u*v] = E[u]*E[v]
    mean_uv = p_u * p_v
    p_est = _spitz_closed_form(p_u, p_v, mean_uv)
    assert np.isclose(p_est, 0.0, atol=1e-10)

    # Test with DEMCalibrator: uncorrelated random syndromes
    rng = np.random.default_rng(42)
    syn = rng.binomial(1, 0.03, size=(5000, 24)).astype(np.uint8)
    calibrator = DEMCalibrator()
    # When uncorrelated and smoothing=0, estimated internal edge probabilities are near min_probability
    calibrated = calibrator.calibrate_from_syndromes(syn, smoothing=0.0)
    internal_probs = [e.probability for e in calibrated.edges if e.detector_a >= 0 and e.detector_b >= 0]
    assert np.mean(internal_probs) < 0.01


def test_spitz_max_covariance_gives_high_p():
    """Perfectly correlated detectors (max covariance) yield estimated edge probability near 0.5."""
    p_corr = 0.49
    # When u and v fire together identically:
    mean_u = p_corr
    mean_v = p_corr
    mean_uv = p_corr
    p_est = _spitz_closed_form(mean_u, mean_v, mean_uv)
    assert p_est > 0.45
    assert p_est < 0.50

    # With PyMatching graph under correlated syndromes
    m = pymatching.Matching()
    m.add_edge(0, 1, fault_ids={0}, weight=5.0, error_probability=0.01)
    shots = 5000
    rng = np.random.default_rng(42)
    flips = rng.binomial(1, 0.48, size=shots)
    syn = np.column_stack([flips, flips])
    m_cal = calibrate_spitz_matching(m, syn)
    edges = list(m_cal.edges())
    assert len(edges) == 1
    est_p = edges[0][2]["error_probability"]
    assert est_p > 0.40


def test_spitz_formula_matches_analytic():
    """For a 2-detector scenario with known injected p_edge, Spitz matches closed-form expectation."""
    pu, pv, pe = 0.02, 0.03, 0.06
    # Exact joint distributions:
    # u is flipped if Eu ^ Euv = 1, v is flipped if Ev ^ Euv = 1
    mu = pu * (1.0 - pe) + (1.0 - pu) * pe
    mv = pv * (1.0 - pe) + (1.0 - pv) * pe
    xuv = pe * (1.0 - pu) * (1.0 - pv) + (1.0 - pe) * pu * pv

    recovered_p = _spitz_closed_form(mu, mv, xuv)
    assert np.isclose(recovered_p, pe, atol=1e-12)


def test_spitz_boundary_estimator_parity_product():
    """Verify parity product identity: 1 - 2*mean = (1 - 2*pb) * prod(1 - 2*pe)."""
    pb_true = 0.025
    pe1_true = 0.018
    pe2_true = 0.022

    prod_e = (1.0 - 2.0 * pe1_true) * (1.0 - 2.0 * pe2_true)
    mean_detector = 0.5 * (1.0 - (1.0 - 2.0 * pb_true) * prod_e)

    # Invert to recover pb
    pb_recovered = 0.5 * (1.0 - (1.0 - 2.0 * mean_detector) / prod_e)
    assert np.isclose(pb_recovered, pb_true, atol=1e-14)


def test_spitz_finite_sample_variance_inflates_at_small_w():
    """W=50 vs W=2000: finite-sample variance of the Spitz estimator inflates significantly at small W."""
    circuit = stim.Circuit(
        "X_ERROR(0.04) 0\n"
        "M 0\n"
        "DETECTOR(0, 0) rec[-1]\n"
        "X_ERROR(0.04) 0\n"
        "M 0\n"
        "DETECTOR(1, 0) rec[-1] rec[-2]\n"
    )
    sampler = circuit.compile_detector_sampler()

    n_trials = 30
    estimates_small = []
    estimates_large = []

    for _ in range(n_trials):
        syn_small = sampler.sample(shots=50)
        u_s = float(syn_small[:, 0].mean())
        v_s = float(syn_small[:, 1].mean())
        uv_s = float((syn_small[:, 0] * syn_small[:, 1]).mean())
        estimates_small.append(_spitz_closed_form(u_s, v_s, uv_s))

        syn_large = sampler.sample(shots=2000)
        u_l = float(syn_large[:, 0].mean())
        v_l = float(syn_large[:, 1].mean())
        uv_l = float((syn_large[:, 0] * syn_large[:, 1]).mean())
        estimates_large.append(_spitz_closed_form(u_l, v_l, uv_l))

    var_small = float(np.var(estimates_small))
    var_large = float(np.var(estimates_large))

    assert var_large < var_small
    assert var_small > 5.0 * var_large
