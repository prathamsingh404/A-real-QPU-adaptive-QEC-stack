"""
Configuration system for AdaptiveQEC.

Pydantic-based typed configuration with YAML loading and environment
variable overrides. The same experiment config can target IBM, IQM,
a simulator, or another architecture without rewriting code.

Credential management:
    1. Place credentials in a .env file (gitignored) at project root.
    2. Or export them as environment variables before running.

Environment variable override convention:
    AQEC_HARDWARE__PROVIDER=ibm
    AQEC_QEC__DISTANCE=5
    AQEC_RUNTIME__GPU__ENABLED=true

Double underscore (__) separates nested config levels.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HardwareProvider(str, Enum):
    IBM = "ibm"
    IQM = "iqm"
    QUANTINUUM = "quantinuum"
    SIMULATOR = "simulator"
