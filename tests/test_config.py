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
