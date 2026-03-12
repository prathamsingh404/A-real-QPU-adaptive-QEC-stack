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
