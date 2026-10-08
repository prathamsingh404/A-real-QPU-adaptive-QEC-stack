"""Tests for the configuration system."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from adaptive_qec.config import (
    AdaptiveQECConfig,
    HardwareProvider,
    QECCode,
    default_config,
    load_config,
    load_config_from_dict,
)


class TestDefaultConfig:
    """Test default configuration creation."""

    def test_default_config_creates(self):
        config = default_config()
        assert config.hardware.provider == HardwareProvider.IBM
        assert config.qec.code == QECCode.SURFACE
        assert config.qec.distance == 3
        assert config.decoder.baseline.value == "mwpm"

    def test_default_config_validates(self):
        config = default_config()
        assert config.qec.distance % 2 == 1  # must be odd
        assert 0 < config.analysis.confidence_level < 1


class TestConfigLoading:
    """Test YAML config loading."""

    def test_load_valid_yaml(self, tmp_path):
        config_data = {
            "hardware": {"provider": "ibm", "backend": "ibm_test"},
            "qec": {"code": "surface", "distance": 5, "rounds": 10},
        }
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_path)
        assert config.hardware.backend == "ibm_test"
        assert config.qec.distance == 5
        assert config.qec.rounds == 10

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_load_partial_yaml(self, tmp_path):
        """Partial configs should use defaults for missing fields."""
        config_data = {"qec": {"distance": 7}}
        config_path = tmp_path / "partial.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_path)
        assert config.qec.distance == 7
        assert config.hardware.provider == HardwareProvider.IBM  # default

    def test_load_from_dict(self):
        config = load_config_from_dict({
            "qec": {"code": "repetition", "distance": 3, "rounds": 5},
        })
        assert config.qec.code == QECCode.REPETITION
        assert config.qec.rounds == 5


class TestConfigValidation:
    """Test configuration validation."""

    def test_even_distance_rejected(self):
        with pytest.raises(Exception):
            load_config_from_dict({"qec": {"distance": 4}})

    def test_zero_rounds_rejected(self):
        with pytest.raises(Exception):
            load_config_from_dict({"qec": {"rounds": 0}})

    def test_invalid_confidence_rejected(self):
        with pytest.raises(Exception):
            load_config_from_dict({"analysis": {"confidence_level": 1.5}})

    def test_negative_noise_rejected(self):
        with pytest.raises(Exception):
            load_config_from_dict({
                "noise": {"gate": {"single_qubit": -0.1}}
            })


class TestEnvOverrides:
    """Test environment variable overrides."""

    def test_env_override_string(self, tmp_path):
        config_path = tmp_path / "base.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"hardware": {"backend": "original"}}, f)

        os.environ["AQEC_HARDWARE__BACKEND"] = "ibm_override"
        try:
            config = load_config(config_path)
            assert config.hardware.backend == "ibm_override"
        finally:
            del os.environ["AQEC_HARDWARE__BACKEND"]

    def test_env_override_int(self, tmp_path):
        config_path = tmp_path / "base.yaml"
        with open(config_path, "w") as f:
            yaml.dump({}, f)

        os.environ["AQEC_QEC__DISTANCE"] = "7"
        try:
            config = load_config(config_path)
            assert config.qec.distance == 7
        finally:
            del os.environ["AQEC_QEC__DISTANCE"]
