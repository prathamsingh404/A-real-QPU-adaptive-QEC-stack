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
    strength: float = Field(default=0.001, ge=0.0, le=1.0)


class LeakageConfig(BaseModel):
    """Leakage noise parameters."""
    enabled: bool = False
    rate: float = Field(default=0.001, ge=0.0, le=1.0)
    seepage: float = Field(default=0.01, ge=0.0, le=1.0)


class DriftConfig(BaseModel):
    """Noise drift parameters."""
    enabled: bool = True
    model: DriftModel = DriftModel.LINEAR
    rate: float = Field(default=0.0001, ge=0.0)
    monitoring_window: int = Field(default=100, ge=1)


class NoiseConfig(BaseModel):
    """Complete noise configuration."""
    readout: ReadoutNoiseConfig = Field(default_factory=ReadoutNoiseConfig)
    gate: GateNoiseConfig = Field(default_factory=GateNoiseConfig)
    correlated: CorrelatedNoiseConfig = Field(default_factory=CorrelatedNoiseConfig)
    leakage: LeakageConfig = Field(default_factory=LeakageConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)


class QECConfig(BaseModel):
    """QEC experiment configuration."""
    code: QECCode = QECCode.SURFACE
    distance: int = Field(default=3, ge=1)
    rounds: int = Field(default=3, ge=1)
    logical_basis: LogicalBasis = LogicalBasis.Z
    boundary: Boundary = Boundary.PLANAR

    @field_validator("distance")
    @classmethod
    def distance_must_be_odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError(f"Code distance must be odd, got {v}")
        return v


class MLDecoderConfig(BaseModel):
    """ML decoder configuration."""
    enabled: bool = False
    model: MLModel = MLModel.CNN
    checkpoint: Optional[str] = None
    batch_size: int = Field(default=256, ge=1)


class AdaptiveDecoderConfig(BaseModel):
    """Adaptive decoder configuration."""
    enabled: bool = False
    noise_estimation: bool = True
    routing: bool = False
    uncertainty: bool = False


class DecoderConfig(BaseModel):
    """Complete decoder configuration."""
    baseline: DecoderType = DecoderType.MWPM
    ml: MLDecoderConfig = Field(default_factory=MLDecoderConfig)
    adaptive: AdaptiveDecoderConfig = Field(default_factory=AdaptiveDecoderConfig)


class CPUConfig(BaseModel):
    """CPU runtime configuration."""
    threads: int = Field(default=4, ge=1)
    affinity: bool = False


class GPUConfig(BaseModel):
    """GPU runtime configuration."""
    enabled: bool = False
    device: int = Field(default=0, ge=0)
    backend: GPUBackendType = GPUBackendType.PYTORCH


class LatencyBudgetConfig(BaseModel):
    """Latency budget configuration for real-time QEC."""
    deadline_us: float = Field(default=1000.0, gt=0)
    warning_threshold: float = Field(default=0.8, gt=0.0, le=1.0)
    tracking: bool = True


class RuntimeConfig(BaseModel):
    """Complete runtime configuration."""
    cpu: CPUConfig = Field(default_factory=CPUConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    batch_size: int = Field(default=1000, ge=1)
    latency_budget: LatencyBudgetConfig = Field(default_factory=LatencyBudgetConfig)


class ExperimentConfig(BaseModel):
    """Experiment management configuration."""
    output_dir: str = "experiments"
    shots: int = Field(default=10000, ge=1)
    repetitions: int = Field(default=1, ge=1)
    save_raw: bool = True
    save_circuits: bool = True
    save_calibration: bool = True
    save_plots: bool = True
    name_template: str = "{code}_d{distance}_r{rounds}_{backend}_{timestamp}"


class DataConfig(BaseModel):
    """Data storage configuration."""
    storage_backend: StorageBackend = StorageBackend.FILESYSTEM
    base_path: str = "experiments"
    compression: bool = False
    format: DataFormat = DataFormat.JSON


class AnalysisConfig(BaseModel):
    """Statistical analysis configuration."""
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    min_shots: int = Field(default=1000, ge=1)
    bootstrap_samples: int = Field(default=10000, ge=100)
    hypothesis_test: HypothesisTest = HypothesisTest.TWO_SIDED


# ---------------------------------------------------------------------------
# Root configuration
# ---------------------------------------------------------------------------

class AdaptiveQECConfig(BaseModel):
    """
    Root configuration for the entire AdaptiveQEC platform.

    Validates the complete configuration tree. Supports loading from YAML
    with environment variable overrides.
    """
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    qec: QECConfig = Field(default_factory=QECConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------

def _apply_env_overrides(data: dict[str, Any], prefix: str = "AQEC") -> dict[str, Any]:
    """
    Apply environment variable overrides to configuration data.

    Convention: AQEC_SECTION__KEY=value
    Double underscore separates nesting levels.
    """
    for key, value in os.environ.items():
        if not key.startswith(f"{prefix}_"):
            continue

        parts = key[len(prefix) + 1:].lower().split("__")
        current = data
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Type coercion for common cases
        raw = value
        if raw.lower() in ("true", "false"):
            raw = raw.lower() == "true"  # type: ignore[assignment]
        else:
            try:
                raw = int(raw)  # type: ignore[assignment]
            except ValueError:
                try:
                    raw = float(raw)  # type: ignore[assignment]
                except ValueError:
                    pass

        current[parts[-1]] = raw

    return data


def _load_dotenv() -> None:
    """Load .env file from project root if it exists."""
    try:
        from dotenv import load_dotenv
        # Walk up from this file to find project root with .env
        here = Path(__file__).resolve().parent
        for ancestor in [here, here.parent, here.parent.parent, here.parent.parent.parent]:
            env_path = ancestor / ".env"
            if env_path.exists():
                load_dotenv(env_path, override=False)
                return
    except ImportError:
