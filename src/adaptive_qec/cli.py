"""
CLI entry point for AdaptiveQEC.

Provides the `aqec` command-line interface registered in pyproject.toml.

Usage:
    aqec run --config configs/default.yaml
    aqec run --config configs/default.yaml --shots 50000
    aqec run --config configs/default.yaml --distances 3,5,7
    aqec serve --port 8000
"""

from __future__ import annotations

import logging
import sys

import click


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
