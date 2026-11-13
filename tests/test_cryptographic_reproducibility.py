"""
Tests for cryptographic seed and config hashing in the reproducibility harness.
"""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from adaptive_qec.config import AQECConfig, load_config
from adaptive_qec.controller.static import StaticController
from adaptive_qec.experiment.harness import ExperimentHarness
from adaptive_qec.experiment.reproducibility import (
    CryptographicManifest,
    ManifestVerificationError,
    generate_cryptographic_manifest,
    hash_file_sha256,
    hash_seeds,
    verify_cryptographic_manifest,
)


def test_hash_file_sha256_and_detection(tmp_path: Path):
    """Verify SHA-256 computation on config files and tamper detection."""
    cfg = tmp_path / "test_config.yaml"
    cfg.write_text("experiment:\n  shots: 1000\nqec:\n  distance: 3\n", encoding="utf-8")

    digest1 = hash_file_sha256(cfg)
    expected = hashlib.sha256(cfg.read_bytes()).hexdigest()
    assert digest1 == expected

    # Tamper with the file
    cfg.write_text("experiment:\n  shots: 2000\nqec:\n  distance: 3\n", encoding="utf-8")
    digest2 = hash_file_sha256(cfg)
    assert digest2 != digest1


def test_hash_seeds_deterministic():
    """Verify deterministic SHA-256 hashing of seeds."""
    seeds1 = [1000, 1001, 1002]
    seeds2 = [1000, 1001, 1002]
    seeds3 = [1000, 1001, 1003]

    h1 = hash_seeds(seeds1)
    h2 = hash_seeds(seeds2)
    h3 = hash_seeds(seeds3)

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64  # SHA-256 hex length


def test_generate_and_verify_cryptographic_manifest(tmp_path: Path):
    """Verify full manifest generation and successful cryptographic verification."""
    cfg1 = tmp_path / "config_main.yaml"
    cfg1.write_text("qec:\n  distance: 3\n  rounds: 3\n", encoding="utf-8")
    cfg2 = tmp_path / "config_noise.yaml"
    cfg2.write_text("noise:\n  p_2q: 0.003\n", encoding="utf-8")

    seeds = [1000, 1001, 1002]
    manifest = generate_cryptographic_manifest(
        experiment_id="exp_test_001",
        config_paths=[cfg1, cfg2],
        seeds=seeds,
        metadata={"provenance": "simulation"},
    )

    assert manifest.manifest_version == "2.0.0"
    assert manifest.experiment_id == "exp_test_001"
    assert len(manifest.config_hashes) == 2
    assert manifest.seeds == seeds
    assert manifest.seeds_sha256 == hash_seeds(seeds)
    assert len(manifest.manifest_digest) == 64

    # Verify untampered manifest
    is_valid, errors = verify_cryptographic_manifest(manifest, expected_seeds=seeds)
    assert is_valid
    assert len(errors) == 0


def test_manifest_verification_detects_tampered_config(tmp_path: Path):
    """Verify that tampering with any config file invalidates the manifest."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("param: 42\n", encoding="utf-8")

    seeds = [42]
    manifest = generate_cryptographic_manifest(
        experiment_id="exp_tamper_test",
        config_paths=[cfg],
        seeds=seeds,
    )

    # Modify the config file on disk
    cfg.write_text("param: 999\n", encoding="utf-8")

    is_valid, errors = verify_cryptographic_manifest(manifest)
    assert not is_valid
    assert any("Config hash mismatch" in e for e in errors)

    # Should raise when raise_on_error is True
    with pytest.raises(ManifestVerificationError):
        verify_cryptographic_manifest(manifest, raise_on_error=True)


def test_manifest_verification_detects_tampered_seeds(tmp_path: Path):
    """Verify that altered seeds fail verification."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("param: 42\n", encoding="utf-8")

    manifest = generate_cryptographic_manifest(
        experiment_id="exp_seed_tamper",
        config_paths=[cfg],
        seeds=[100, 200],
    )

    # Tamper with seeds in manifest dict
    m_dict = manifest.to_dict()
    m_dict["seeds"] = [100, 201]

    is_valid, errors = verify_cryptographic_manifest(m_dict)
    assert not is_valid
    assert any("Seeds SHA-256 mismatch" in e for e in errors)


def test_harness_manifest_generation_and_verification(tmp_path: Path):
    """Verify that ExperimentHarness generates, verifies, and persists cryptographic manifests."""
    # Write a test config file
    cfg_file = tmp_path / "default_harness.yaml"
    cfg_dict = {
        "qec": {"distance": 3, "rounds": 3},
        "experiment": {"output_dir": str(tmp_path / "results"), "shots": 100},
        "noise": {
            "gate": {"single_qubit": 0.0005, "two_qubit": 0.003},
            "readout": {"p0_given_1": 0.01, "p1_given_0": 0.01},
        },
    }
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(cfg_dict, f)

    ctrl = StaticController()
    seeds = [101, 102]
    harness = ExperimentHarness.from_config(
        config_path=str(cfg_file),
        controllers=[ctrl],
        output_dir=tmp_path / "results",
        seeds=seeds,
    )

    # Generate and verify manifest directly from harness
    manifest = harness.generate_manifest(experiment_id="test_run_harness")
    assert str(cfg_file).replace("\\", "/") in manifest.config_hashes
    assert manifest.seeds == seeds

    is_valid, errors = harness.verify_manifest(manifest)
    assert is_valid
    assert len(errors) == 0

    # Run 2 windows and verify saved files and manifest
    res = harness.run(num_windows=2, shots_per_window=50, seeds=seeds, record_manifest=True)
    assert res.manifest is not None
    assert res.manifest["seeds"] == seeds
    assert len(res.manifest["config_hashes"]) > 0

    # Verify saved files exist on disk
    exp_json = tmp_path / "results" / f"{res.experiment_id}.json"
    manifest_json = tmp_path / "results" / f"{res.experiment_id}_manifest.json"
    assert exp_json.exists()
    assert manifest_json.exists()

    with open(manifest_json, "r", encoding="utf-8") as f:
        saved_manifest = json.load(f)
    assert saved_manifest["experiment_id"] == res.experiment_id
    assert saved_manifest["seeds_sha256"] == hash_seeds(seeds)
