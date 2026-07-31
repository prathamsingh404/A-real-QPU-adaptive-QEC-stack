"""
Unit tests for the AdaptiveQEC CLI (aqec).
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from adaptive_qec.cli import cli


class TestCLI:
    """Tests for Click-based CLI entry points."""

    def test_cli_help(self) -> None:
        """Test aqec --help exits with code 0."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "AdaptiveQEC" in result.output
        assert "Commands:" in result.output

    def test_cli_version(self) -> None:
        """Test aqec --version prints the correct version."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.2.0" in result.output

    def test_cli_check_default(self) -> None:
        """Test aqec check on default config."""
        runner = CliRunner()
        result = runner.invoke(cli, ["check", "--config", "configs/default.yaml"])
        # If token is set in env or missing, check handles it cleanly
        assert "Config valid:" in result.output

    def test_cli_check_nonexistent(self) -> None:
        """Test aqec check with nonexistent config path."""
        runner = CliRunner()
        result = runner.invoke(cli, ["check", "--config", "configs/nonexistent_xyz.yaml"])
        assert result.exit_code != 0
        assert "[ERROR]" in result.output

    def test_cli_benchmark_decoder(self) -> None:
        """Test aqec benchmark-decoder command with small shot count."""
        runner = CliRunner()
        result = runner.invoke(cli, ["benchmark-decoder", "--shots", "100", "--distance", "3", "--rounds", "2"])
        assert result.exit_code == 0
        assert "DECODER BENCHMARK RESULTS" in result.output
        assert "Union-Find:" in result.output
        assert "MWPM:" in result.output
