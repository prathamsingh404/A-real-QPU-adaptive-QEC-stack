"""Causal recalibration of decomposed DEM fault priors prior to correlated matching."""
import numpy as np
import pymatching
import stim
from scipy.sparse import csr_matrix


class CorrelatedPrior:
    def __init__(self, circuit):
        self.dem = circuit.detector_error_model(decompose_errors=True).flattened()
        self.instructions = list(self.dem)
        self.faults = []
        detectors, mechanisms, probabilities = [], [], []
        for i, inst in enumerate(self.instructions):
            if inst.type != "error":
                continue
            # A detector repeated across suggested components cancels under XOR.
            support = set()
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    support.symmetric_difference_update([t.val])
            detectors.extend(sorted(support))
            mechanisms.extend([len(self.faults)] * len(support))
            probabilities.append(inst.args_copy()[0])
            self.faults.append(i)
        self.incidence = csr_matrix((np.ones(len(detectors)), (detectors, mechanisms)),
                                    shape=(self.dem.num_detectors,len(self.faults)))
        self.degree = np.maximum(1, self.incidence.sum(axis=0).A1)
        self.prior = np.array(probabilities)
        if np.any(self.prior >= 0.5):
            raise ValueError("Parity attenuation model requires priors below 0.5")
        self.attenuation = -np.log1p(-2*self.prior)
        self.expected = self.incidence @ self.attenuation

    def rate(self, attenuation):
        return np.clip(-np.expm1(-(self.incidence @ attenuation))/2, 1e-7, 1-1e-7)

    def estimate(self, samples, mode):
        # Prior pseudo-counts regularize weakly observed detector marginals.
        nominal = self.rate(self.attenuation)
        observed = (samples.sum(axis=0) + 200*nominal)/(len(samples)+200)
        target = -np.log1p(-2*np.clip(observed, 1e-7, 0.49))
        residual = np.clip(np.log(target/np.maximum(self.expected,1e-7)), -2,2)
        if mode == "local":
            order = np.argsort(-abs(residual),kind="stable")
            residual[order[max(1,len(order)//8):]] = 0
        elif mode != "full":
            raise ValueError(mode)
        return self.attenuation * np.exp((self.incidence.T @ residual)/self.degree)

    def graph(self, attenuation=None):
        dem = self.dem
        if attenuation is not None:
            probabilities = -np.expm1(-attenuation)/2
            dem = stim.DetectorErrorModel()
            j = 0
            for op in self.instructions:
                if op.type == "error":
                    dem.append("error",float(probabilities[j]),op.targets_copy())
                    j += 1
                else:
                    dem.append(op)
        return pymatching.Matching.from_detector_error_model(dem,enable_correlations=True)

    def choose(self, samples, penalty):
        training, validation = np.array_split(samples,2)
        counts = validation.sum(axis=0)
        def loss(a):
            p = self.rate(a)
            return -float(counts @ np.log(p) + (len(validation)-counts) @ np.log1p(-p))
        candidates = {"factory":self.attenuation}
        scores = {"factory":loss(self.attenuation)}
        for name, cost in [("local",1),("full",2)]:
            candidates[name] = self.estimate(training,name)
            scores[name] = loss(candidates[name]) + penalty*cost
        selected = min(scores,key=scores.get)
        return selected, None if selected == "factory" else candidates[selected]
