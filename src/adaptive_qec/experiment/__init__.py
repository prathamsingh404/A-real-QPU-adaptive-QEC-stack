"""Experiment management — orchestration and reproducibility."""

from adaptive_qec.experiment.distance_sweep import (
    DistanceSweep,
    DistanceSweepResults,
    SweepResult,
)
from adaptive_qec.experiment.harness import (
    ExperimentHarness,
    ExperimentRunResult,
    WindowResult,
)
from adaptive_qec.experiment.reproducibility import (
    CryptographicManifest,
    ManifestVerificationError,
    capture_reproducibility_info,
    generate_cryptographic_manifest,
    hash_file_sha256,
    hash_seeds,
    verify_cryptographic_manifest,
)

__all__ = [
    "DistanceSweep",
    "DistanceSweepResults",
    "SweepResult",
    "ExperimentHarness",
    "ExperimentRunResult",
    "WindowResult",
    "CryptographicManifest",
    "ManifestVerificationError",
    "capture_reproducibility_info",
    "generate_cryptographic_manifest",
    "hash_file_sha256",
    "hash_seeds",
    "verify_cryptographic_manifest",
]
