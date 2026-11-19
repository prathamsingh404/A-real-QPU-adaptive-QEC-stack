"""Check the independent timing experiment against its trained schedule and physical controls."""
import json
from pathlib import Path
import numpy as np
import pytest
import yaml

from adaptive_qec.analysis.causal_results import contrast


def read(name):
    return json.loads(Path("data", name+".json").read_text())


def test_confirmation_uses_disjoint_seeds_and_one_update():
    history = read("suite_results")["trajectories"]
    settings = yaml.safe_load(Path("configs/timing_confirmation.yaml").read_text())
    outcome = read("timing_confirmation")
    training = read("timing_development")
    old_seeds = {r["seed"] for r in history}
    for panel in settings["panels"]:
        d = str(panel["distance"])
        result = outcome["panels"][d]
        records = result["records"]
        seeds = {r["seed"] for r in records}
        assert seeds == set(range(panel["first_seed"],panel["first_seed"]+panel["count"]))
        assert not seeds & old_seeds
        look = 1 + np.argmin(training[d]["totals"])
        assert look == training[d]["best_look"] == result["fixed_look"]
        assert all(r["selected_look"] == (r["first_alarm_look"] or len(r["candidate_errors"])) for r in records)
        assert all(r["selective_errors"] == r["candidate_errors"][r["selected_look"]-1] for r in records)
        differences = [(r["selective_errors"]-r["candidate_errors"][look-1])/12000 for r in records]
        assert contrast(differences)["mean"] == pytest.approx(result["paired"]["mean"])


def test_independent_physical_replay_factory_and_publisher_agree():
    additional = read("public_decoder_comparison")
    original = read("public_replay_results")
    for panel in additional["records"]:
        first = next(r for r in original["records"] if r["basis"] == panel["basis"])
        assert panel["shots"] == first["shots"]
        assert panel["errors"]["local_uncorrelated"] == sum(first["errors"]["static"])
        assert panel["errors"]["correlated_matching_decoder_with_si1000_prior"] == sum(first["errors"]["publisher_correlated"])
