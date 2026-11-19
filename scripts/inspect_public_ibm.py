"""Inspect third-party Fez repetition-memory registers and score terminal majority on all d=3, r=1 snapshots."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

SOURCE = "https://zenodo.org/records/20768087/files/ibm_fez.tar.gz"
PUBLISHED_MD5 = "27ce2827dde0713a436179121bfd7173"


def inspect(archive):
    if hashlib.md5(archive.read_bytes()).hexdigest() != PUBLISHED_MD5:
        raise ValueError("Archive differs from the publisher's checksum")
    rows = []
    with tarfile.open(archive) as t:
        metadata = sorted(n for n in t.getnames() if n.endswith("/info.json"))
        for name in metadata:
            info = json.load(t.extractfile(name))
            if (info["d"],info["rounds"]) != (3,1):
                continue
            samples = json.load(t.extractfile(name.replace("info.json","bitstrings.json")))
            for circuit in samples:
                truth = circuit["metadata"]["logical_state"]
                regs = circuit["per_shot_cregs"]
                errors = {}
                for register, shots in regs.items():
                    if not register.startswith("c_data_"):
                        continue
                    if len(shots)!=info["shots"] or any(len(x)!=3 or any(b not in (0,1) for b in x) for x in shots):
                        raise ValueError("Unexpected terminal register layout")
                    errors[register] = sum((sum(bits)>=2) != bool(truth) for bits in shots)
                if len(errors)!=info["n_chains"]:
                    raise ValueError("Missing chain register")
                rows.append(dict(snapshot=name, basis=info["basis"], prepared_state=truth,
                                 device_shots=info["shots"], parallel_chains=len(errors), errors_per_chain=errors))
    return dict(source=SOURCE, archive_md5=PUBLISHED_MD5, device="ibm_fez", archive_snapshots=len(metadata), records=rows,
                scope="External data-format validation. Terminal majority compared with known preparation. Chains within a device shot share an acquisition.")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path)
    p.add_argument("--output",type=Path,default=Path("data/external_ibm_majority.json"))
    args=p.parse_args()
    result=inspect(args.archive)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8",newline="\n")
    print([(r["basis"],r["prepared_state"],r["device_shots"],r["parallel_chains"],sum(r["errors_per_chain"].values())) for r in result["records"]])
