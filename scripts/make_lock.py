"""Freeze the files used for the paper. Text digests use LF on every platform."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compute_sha256(path):
    data = path.read_bytes()
    if path.suffix.lower() not in {".b8", ".pdf", ".png"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def paper_files():
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    fixed = {*manifest["source_sha256"], "configs/suite.yaml", "data/manifest.json", "data/suite_results.json",
             "data/analysis.json", "data/midpoint_results.json", "scripts/check_single_fit.py", "data/host_timings.json", "data/public_replay_results.json", "data/public_alignment_results.json",
             "README.md", "VALIDATION.md", "papers/main.tex", "papers/main.pdf", "CITATION.cff", "uv.lock",
             "scripts/make_claims.py", "scripts/make_lock.py", "scripts/verify_lock.py", "scripts/reproduce.py",
             "scripts/profile_reweighting.py", "scripts/replay_public_memory.py", "scripts/replay_alignment.py",
             "src/adaptive_qec/analysis/causal_results.py", "src/adaptive_qec/analysis/make_paper_tables.py",
             "src/adaptive_qec/analysis/make_paper_figures.py", "docs/prior_art_matrix.md",
             "papers/figures/paired_comparison.pdf", "papers/figures/paired_comparison.png"}
    fixed.update({"TIMING_PROTOCOL.md", "configs/timing_confirmation.yaml", "data/timing_manifest.json",
                  "data/timing_development.json", "data/timing_confirmation.json", "scripts/confirm_update_timing.py",
                  "reports/REBUILD_REPORT.md", "scripts/inspect_public_ibm.py", "data/external_ibm_majority.json", "scripts/check_ising_sample_format.py", "docs/implementation_review.md", "scripts/compare_public_decoders.py", "data/public_decoder_comparison.json"})
    fixed.update(f"papers/tables/table_{i}.tex" for i in range(1,6))
    fixed.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "data/public_replay").rglob("*") if p.is_file())
    return sorted(fixed)


def check_manifest():
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    targets = {**manifest["source_sha256"], "configs/suite.yaml":manifest["config_sha256"],
               "data/suite_results.json":manifest["results_file"]["sha256"]}
    follow = json.loads((ROOT / "data/timing_manifest.json").read_text())
    targets.update(follow["source_sha256"])
    targets["data/timing_confirmation.json"] = follow["result_sha256"]
    for name, expected in targets.items():
        if compute_sha256(ROOT / name) != expected:
            raise ValueError(f"Frozen simulation digest changed: {name}")
    return manifest


if __name__ == "__main__":
    manifest = check_manifest()
    hashes = {name:compute_sha256(ROOT / name) for name in paper_files()}
    meta = dict(version=3, provenance="simulation, public physical replay, host profiling",
                hash_algorithm="SHA-256", git_commit=manifest["git_commit"], git_commit_scope="pre-evaluation simulation",
                created_at_utc=datetime.now(timezone.utc).isoformat(), total_tracked_files=len(hashes))
    (ROOT / "paper.lock").write_text(json.dumps(dict(metadata=meta, files=hashes), indent=2)+"\n", encoding="utf8", newline="\n")
