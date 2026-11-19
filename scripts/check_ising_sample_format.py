"""Check the NVIDIA sparse writer against Google's physical bits. Pass the pinned upstream module."""
import argparse
import hashlib
import importlib.util
import tempfile
from pathlib import Path
import numpy as np
import stim
from replay_public_memory import DATA, read_bits


def check(path):
    source = path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(source).hexdigest() != EXPECTED:
        raise ValueError("Supply stim_sample_io.py from the pinned NVIDIA revision")
    spec = importlib.util.spec_from_file_location("ising_io",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as tmp:
        for basis in ("X","Z"):
            folder=DATA/basis/"r13"
            mem=stim.Circuit.from_file(folder/"circuit_noisy_si1000.stim")
            widths=dict(num_detectors=mem.num_detectors,num_observables=mem.num_observables)
            x=read_bits(folder/"detection_events.b8",50000,mem.num_detectors)[6000:]
            y=read_bits(folder/"obs_flips_actual.b8",50000,mem.num_observables)[6000:]
            expected=np.concatenate((x,y),axis=1)
            target=Path(tmp)/(basis+".dets")
            module.write_stim_detector_samples(path=target,dets_and_obs=expected,**widths)
            actual=stim.read_shot_data_file(path=target,format="dets",**widths)
            if not np.array_equal(actual,expected):
                raise ValueError(f"Physical bits changed for basis {basis}")
            print(basis,expected.shape,"NVIDIA sparse writer round trip passed")


EXPECTED = "2d5ee73af71c986c805e59fbcb8ceb07ce8ff6d9345220db0177539b6debf4de"


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream_module",type=Path)
    check(parser.parse_args().upstream_module)
