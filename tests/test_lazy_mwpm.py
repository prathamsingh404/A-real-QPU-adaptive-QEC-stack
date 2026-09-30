import pytest
import numpy as np
import stim
from adaptive_qec.decoders.lazy_mwpm import LazyMWPMDecoder
from adaptive_qec.decoders.registry import get_decoder

def test_lazy_mwpm_initialization():
    dec = LazyMWPMDecoder(defect_threshold=10)
    assert dec.name == 'lazy_mwpm'
    assert dec.defect_threshold == 10

def test_lazy_mwpm_registry():
    dec = get_decoder('lazy_mwpm', defect_threshold=12)
    assert isinstance(dec, LazyMWPMDecoder)
    assert dec.defect_threshold == 12

def test_lazy_mwpm_configuration():
    circuit = stim.Circuit.generated('surface_code:rotated_memory_z', distance=3, rounds=3, after_clifford_depolarization=0.01)
    dem = circuit.detector_error_model()
    dec = LazyMWPMDecoder(defect_threshold=5)
    dec.configure(circuit=circuit, dem=dem)
    assert dec._num_detectors > 0
    assert dec._num_observables > 0

def test_lazy_mwpm_zero_syndrome():
    circuit = stim.Circuit.generated('surface_code:rotated_memory_z', distance=3, rounds=3, after_clifford_depolarization=0.01)
    dem = circuit.detector_error_model()
    dec = LazyMWPMDecoder(defect_threshold=5)
    dec.configure(circuit=circuit, dem=dem)
    syn0 = np.zeros(dec._num_detectors, dtype=np.uint8)
    corr0 = dec.decode(syn0)
    assert corr0.observable_corrections.shape[-1] == dec._num_observables
    assert np.all(corr0.observable_corrections == 0)

def test_lazy_mwpm_sparse_syndrome():
    circuit = stim.Circuit.generated('surface_code:rotated_memory_z', distance=3, rounds=3, after_clifford_depolarization=0.01)
    dem = circuit.detector_error_model()
    dec = LazyMWPMDecoder(defect_threshold=5)
    dec.configure(circuit=circuit, dem=dem)
    syn_sparse = np.zeros(dec._num_detectors, dtype=np.uint8)
    syn_sparse[0] = 1
    corr_sparse = dec.decode(syn_sparse)
    assert corr_sparse.observable_corrections is not None

def test_lazy_mwpm_high_density_routing():
    circuit = stim.Circuit.generated('surface_code:rotated_memory_z', distance=3, rounds=3, after_clifford_depolarization=0.01)
    dem = circuit.detector_error_model()
    dec = LazyMWPMDecoder(defect_threshold=2)
    dec.configure(circuit=circuit, dem=dem)
    syn_high = np.ones(dec._num_detectors, dtype=np.uint8)
    assert dec.is_ambiguous(syn_high)
    corr_high = dec.decode(syn_high)
    assert corr_high is not None
