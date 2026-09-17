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
    model_config = {"arbitrary_types_allowed": True}

    provider: HardwareProvider = HardwareProvider.IBM
    backend: str = "ibm_marrakesh"
    qubits: int = Field(default=156, ge=1)
    topology: str = "heavy_hex"
    api_token_env: str = "IBM_QUANTUM_TOKEN"
    channel: str = "ibm_cloud"
    instance_env: str = "IBM_QUANTUM_INSTANCE"

    @property
    def api_token(self) -> Optional[str]:
        """Retrieve API token from environment variable."""
        return os.environ.get(self.api_token_env)

    @property
    def instance(self) -> Optional[str]:
        """Retrieve CRN/instance from environment variable."""
        return os.environ.get(self.instance_env)


class ReadoutNoiseConfig(BaseModel):
    """Readout noise parameters."""
    enabled: bool = True
    p0_given_1: float = Field(default=0.01208, ge=0.0, le=1.0)
    p1_given_0: float = Field(default=0.01208, ge=0.0, le=1.0)


class GateNoiseConfig(BaseModel):
    """Gate noise parameters."""
    single_qubit: float = Field(default=0.000454, ge=0.0, le=1.0)
    two_qubit: float = Field(default=0.003021, ge=0.0, le=1.0)
    model: NoiseModel = NoiseModel.DEPOLARIZING


class CorrelatedNoiseConfig(BaseModel):
    """Correlated noise parameters."""
    enabled: bool = False
    spatial_range: int = Field(default=1, ge=0)
