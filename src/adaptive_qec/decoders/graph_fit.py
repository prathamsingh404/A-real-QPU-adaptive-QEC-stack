"""Fit graphlike matching weights from a batch of detector samples."""
import numpy as np
import pymatching

_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)


class SpitzGraph:
    """Cache the factory topology. Estimate interior edges before inverting boundary parities."""

    def __init__(self, factory):
        self.edges = list(factory.edges())
        self.prior = np.array([d["error_probability"] for _, _, d in self.edges])
        self.bulk = np.array([i for i, (_, v, _) in enumerate(self.edges) if v is not None], dtype=int)
        self.boundary = np.array([i for i, (_, v, _) in enumerate(self.edges) if v is None], dtype=int)
        self.u = np.array([self.edges[i][0] for i in self.bulk], dtype=int)
        self.v = np.array([self.edges[i][1] for i in self.bulk], dtype=int)
        self.b = np.array([self.edges[i][0] for i in self.boundary], dtype=int)
        self.n = factory.num_detectors

    def estimate(self, samples):
        if samples.ndim != 2 or samples.shape[1] != self.n or not len(samples):
            raise ValueError("Expected a nonempty shot-by-detector matrix for the factory graph")
        means = samples.mean(axis=0)
        packed = np.packbits(samples.T, axis=1)
        joint = _POPCOUNT[packed[self.u] & packed[self.v]].sum(axis=1) / len(samples)
        num = 4 * (joint - means[self.u] * means[self.v])
        den = 1 - 2 * (means[self.u] + means[self.v]) + 4 * joint
        ratio = np.divide(num, den, out=np.zeros_like(num), where=np.abs(den) > 1e-9)
        weights = np.empty(len(self.edges))
        internal = np.clip((1 - np.sqrt(np.maximum(1 - ratio, 0))) / 2, 1e-4, 0.49)
        weights[self.bulk] = internal
        product = np.ones(self.n)
        np.multiply.at(product, self.u, 1 - 2 * internal)
        np.multiply.at(product, self.v, 1 - 2 * internal)
        boundary = np.divide(1 - 2 * means[self.b], product[self.b],
                             out=np.ones(len(self.b)), where=product[self.b] > 1e-9)
        weights[self.boundary] = np.clip((1 - boundary) / 2, 1e-4, 0.49)
        return weights

    def matching(self, estimate, prior_fraction):
        probabilities = prior_fraction * self.prior + (1 - prior_fraction) * estimate
        graph = pymatching.Matching()
        for (u, v, data), p, w in zip(self.edges, probabilities, np.log((1 - probabilities) / probabilities)):
            attrs = dict(weight=w, error_probability=p, fault_ids=data["fault_ids"])
            if v is None:
                graph.add_boundary_edge(u, **attrs)
            else:
                graph.add_edge(u, v, **attrs)
        return graph
