"""Score public predictions and run correlated PyMatching on identical physical shots."""
import json
import numpy as np
import pymatching
import stim
import yaml
if __package__:
    from .replay_public_memory import ROOT, DATA, read_bits
else:
    from replay_public_memory import ROOT, DATA, read_bits


def compare():
    ref = yaml.safe_load((ROOT/"configs/suite.yaml").read_text())["reference_shots"]
    results = []
    for basis in ("X","Z"):
        folder = DATA/basis/"r13"
        n = json.loads((folder/"metadata.json").read_text())["shots"]
        mem = stim.Circuit.from_file(folder/"circuit_noisy_si1000.stim")
        det = read_bits(folder/"detection_events.b8",n,mem.num_detectors)[ref:]
        truth = read_bits(folder/"obs_flips_actual.b8",n,mem.num_observables)[ref:]
        dem = mem.detector_error_model(decompose_errors=True)
        errors = {}
        for correlations in (False,True):
            graph = pymatching.Matching.from_detector_error_model(dem,enable_correlations=correlations)
            pred = graph.decode_batch(det,enable_correlations=correlations)
            errors["local_correlated" if correlations else "local_uncorrelated"] = int(np.count_nonzero(pred!=truth))
        for path in sorted((folder/"decoding_results").glob("*/obs_flips_predicted.b8")):
            pred = read_bits(path,n,mem.num_observables)[ref:]
            errors[path.parent.name] = int(np.count_nonzero(pred!=truth))
        results.append(dict(basis=basis, first_scored_row=ref, shots=len(truth), errors=errors))
    return dict(records=results, source="https://zenodo.org/records/13273331",
                scope="Offline replay. Publisher RL priors used 13-cycle calibration data; independence from these records is unverified.")


if __name__ == "__main__":
    outcome=compare()
    (ROOT/"data/public_decoder_comparison.json").write_text(json.dumps(outcome,indent=2)+"\n",encoding="utf8",newline="\n")
    print(outcome["records"])
