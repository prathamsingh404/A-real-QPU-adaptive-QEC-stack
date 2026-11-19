"""Select a precompiled single-qubit fault model for the next memory batch."""
import numpy as np
import stim

from adaptive_qec.decoders.correlated_updates import CorrelatedPrior
from adaptive_qec.experiments.causal_reweighting import data_qubits


class FaultLibrary:
    def __init__(self, memory):
        memory = memory.flattened()
        self.graphs = []
        rates = []
        self.targets = [None, *data_qubits(memory)]
        for target in self.targets:
            model = memory
            if target is not None:
                model = stim.Circuit()
                for op in memory:
                    model.append(op)
                    if op.name == "TICK":
                        model.append("DEPOLARIZE1",[target],0.02)
            template = CorrelatedPrior(model)
            self.graphs.append(template.graph())
            rates.append(template.rate(template.attenuation))
        p = np.asarray(rates)
        self.log_odds = np.log(p) - np.log1p(-p)
        self.log_quiet = np.log1p(-p).sum(axis=1)

    def select(self, x):
        scores = self.log_odds @ x.sum(axis=0) + len(x)*self.log_quiet
        choice = int(scores.argmax())
        return choice if scores[choice]-scores[0] > 16 else 0
