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
    MOCK = "mock"


class QECCode(str, Enum):
    REPETITION = "repetition"
    SURFACE = "surface"
    COLOR = "color"
    QLDPC = "qldpc"


class LogicalBasis(str, Enum):
    Z = "Z"
    X = "X"


class Boundary(str, Enum):
    PLANAR = "planar"
    TORIC = "toric"


class NoiseModel(str, Enum):
    DEPOLARIZING = "depolarizing"
    PAULI = "pauli"
    CUSTOM = "custom"


class DriftModel(str, Enum):
    LINEAR = "linear"
    SINUSOIDAL = "sinusoidal"
    RANDOM_WALK = "random_walk"


class DecoderType(str, Enum):
    MWPM = "mwpm"
    UNION_FIND = "union_find"
    BELIEF_PROPAGATION = "belief_propagation"
    OSD = "osd"
    CNN = "cnn"
    GNN = "gnn"
    TRANSFORMER = "transformer"
    ADAPTIVE = "adaptive"
    LOOKUP = "lookup"


class MLModel(str, Enum):
    CNN = "cnn"
    GNN = "gnn"
    TRANSFORMER = "transformer"


class GPUBackendType(str, Enum):
    PYTORCH = "pytorch"
    TENSORRT = "tensorrt"
    CUDA = "cuda"


class StorageBackend(str, Enum):
    FILESYSTEM = "filesystem"
    POSTGRESQL = "postgresql"


class DataFormat(str, Enum):
    JSON = "json"
    PARQUET = "parquet"
    HDF5 = "hdf5"


class HypothesisTest(str, Enum):
    TWO_SIDED = "two_sided"
    ONE_SIDED = "one_sided"


# ---------------------------------------------------------------------------
# Configuration sub-models
# ---------------------------------------------------------------------------

class HardwareConfig(BaseModel):
    """QPU hardware configuration."""
