"""Rescore ordered physical registers. A hardware summary is insufficient input."""
import argparse
import json
from pathlib import Path

import numpy as np
import pymatching

from adaptive_qec.qpu.repetition_records import convert_registers
from adaptive_qec.decoders.union_find import UnionFindDecoder


def replay_hardware_bitstrings(results_path: Path, output_path: Path):
    d=json.loads(results_path.read_text())
    if d.get("provenance")!="ibm_runtime_ordered_registers" or not d.get("job_id"):
        raise ValueError("Raw IBM registers with provider job ID required; aggregate summaries cannot be replayed")
    mem,x,y=convert_registers(d["registers"])
    dem=mem.detector_error_model(decompose_errors=True)
    uf=UnionFindDecoder()
    uf.configure(dem=dem)
    predictions={"mwpm":pymatching.Matching.from_detector_error_model(dem).decode_batch(x),
                 "union_find":uf.decode(x).observable_corrections}
    errors={name:int(np.count_nonzero(p.reshape(y.shape)!=y)) for name,p in predictions.items()}
    errors["majority_only"]=sum(s.count("1")>len(s)//2 for s in d["registers"]["meas_data"])
    result=dict(provenance="offline IBM register replay",job_id=d["job_id"],backend=d["backend"],
                observable="Stim repetition_code:memory L0; known zero input", shots=len(x),errors=errors)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("registers",type=Path)
    ap.add_argument("output",type=Path)
    args=ap.parse_args()
    replay_hardware_bitstrings(args.registers,args.output)
