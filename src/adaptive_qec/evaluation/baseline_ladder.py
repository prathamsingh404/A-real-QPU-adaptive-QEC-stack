"""
The Formal 5-Level Baseline Ladder and Evaluation Hierarchy for Adaptive QEC.

Defines the formalized 5-level hierarchy of baselines:
    Level 0: Static MWPM (Standard non-adaptive baseline)
    Level 1: Periodic Spitz Reweighting (Periodic syndrome re-estimation without gating)
    Level 2: CUSUM-gated (CUSUM change-detection gated reweighting)
    Level 3: Full 3-Gate CG-CCADS (Confidence-Gated Cost-Cognizant Adaptive Decoding System)
    Level 4 / Ceiling: Oracle MWPM (Theoretical ceiling with privileged true noise knowledge)

Also maintains backward-compatible 7-rung mapping (B0-B6) and provides relative
performance delta benchmarking against Level 0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import stim

from adaptive_qec.decoders.base import Decoder
from adaptive_qec.decoders.mwpm import MWPMDecoder
from adaptive_qec.decoders.reweighted_mwpm import OnlineReweightedMWPM
from adaptive_qec.decoders.union_find import UnionFindDecoder
from adaptive_qec.decoders.dem_calibration import DEMCalibrator
from adaptive_qec.evaluation.counterfactual import CounterfactualEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formal 5-Level Baseline Hierarchy
# ---------------------------------------------------------------------------

class BaselineLevel(int, Enum):
    """
    Formal 5-level hierarchy of QEC decoding baselines:
      Level 0: Static MWPM (Standard non-adaptive baseline)
      Level 1: Periodic Spitz Reweighting (Periodic syndrome re-estimation without gating)
      Level 2: CUSUM-gated (CUSUM change-detection gated reweighting)
      Level 3: Full 3-Gate CG-CCADS (Confidence-Gated Cost-Cognizant Adaptive Decoding System)
      Level 4: Oracle MWPM (Theoretical ceiling with privileged true noise knowledge)
    """
    LEVEL_0_STATIC_MWPM = 0
    LEVEL_1_PERIODIC_SPITZ = 1
    LEVEL_2_CUSUM_GATED = 2
    LEVEL_3_FULL_CG_CCADS = 3
    ORACLE_CEILING = 4


class BaselineRung(str, Enum):
    """Legacy 7-rung baseline notation."""
    B0_FACTORY_MWPM = "b0_factory_mwpm"
    B1_OPTIMIZED_MWPM = "b1_optimized_mwpm"
    B2_TRUE_UNION_FIND = "b2_true_union_find"
    B3_ONLINE_REWEIGHTED = "b3_online_reweighted_mwpm"
    B4_NAIVE_ADAPTIVE = "b4_naive_adaptive"
    B5_SELECTIVE_CG_CCADS = "b5_selective_cg_ccads"
    B6_ORACLE = "b6_oracle"


@dataclass
class BaselineLevelInfo:
    """Metadata describing a baseline ladder level."""
    level: BaselineLevel
    short_name: str
    formal_name: str
    description: str
    is_adaptive: bool
    is_oracle: bool
    rung_equivalent: Optional[BaselineRung] = None


BASELINE_LEVEL_REGISTRY: dict[BaselineLevel, BaselineLevelInfo] = {
    BaselineLevel.LEVEL_0_STATIC_MWPM: BaselineLevelInfo(
        level=BaselineLevel.LEVEL_0_STATIC_MWPM,
        short_name="Level 0",
        formal_name="Static MWPM",
        description="Standard non-adaptive MWPM decoder using nominal/factory calibration weights.",
        is_adaptive=False,
        is_oracle=False,
        rung_equivalent=BaselineRung.B0_FACTORY_MWPM,
    ),
    BaselineLevel.LEVEL_1_PERIODIC_SPITZ: BaselineLevelInfo(
        level=BaselineLevel.LEVEL_1_PERIODIC_SPITZ,
        short_name="Level 1",
        formal_name="Periodic Spitz Reweighting",
        description="Sliding-window periodic syndrome defect reweighting without gating or hysteresis (Spitz et al.).",
        is_adaptive=True,
        is_oracle=False,
        rung_equivalent=BaselineRung.B3_ONLINE_REWEIGHTED,
    ),
    BaselineLevel.LEVEL_2_CUSUM_GATED: BaselineLevelInfo(
        level=BaselineLevel.LEVEL_2_CUSUM_GATED,
        short_name="Level 2",
        formal_name="CUSUM-Gated Reweighting",
        description="Adaptation gated strictly by sequential CUSUM drift detection without full cost awareness.",
        is_adaptive=True,
        is_oracle=False,
        rung_equivalent=BaselineRung.B4_NAIVE_ADAPTIVE,
    ),
    BaselineLevel.LEVEL_3_FULL_CG_CCADS: BaselineLevelInfo(
        level=BaselineLevel.LEVEL_3_FULL_CG_CCADS,
        short_name="Level 3",
        formal_name="Full 3-Gate CG-CCADS",
        description="Full confidence-gated, cost-cognizant adaptive decoding system with LCB, deadband hysteresis, and rollback guardrails.",
        is_adaptive=True,
        is_oracle=False,
        rung_equivalent=BaselineRung.B5_SELECTIVE_CG_CCADS,
    ),
    BaselineLevel.ORACLE_CEILING: BaselineLevelInfo(
        level=BaselineLevel.ORACLE_CEILING,
        short_name="Level 4 (Ceiling)",
        formal_name="Oracle MWPM",
        description="Theoretical performance ceiling with privileged instantaneous access to true physical error rates.",
        is_adaptive=False,
        is_oracle=True,
        rung_equivalent=BaselineRung.B6_ORACLE,
    ),
}


# ---------------------------------------------------------------------------
# Relative Delta Benchmarking Helper
# ---------------------------------------------------------------------------

@dataclass
class BaselineDeltaBenchmark:
    """Relative performance delta benchmark against Level 0 (Static MWPM)."""
    level: BaselineLevel
    formal_name: str
    ler: float
    reference_ler: float
    delta_ler: float               # ler - reference_ler (negative indicates improvement)
    relative_delta_pct: float      # (ler - reference_ler) / reference_ler * 100
    relative_reduction_pct: float  # (reference_ler - ler) / reference_ler * 100
    is_improvement: bool           # ler < reference_ler

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "formal_name": self.formal_name,
            "ler": round(self.ler, 6),
            "reference_ler": round(self.reference_ler, 6),
            "delta_ler": round(self.delta_ler, 6),
            "relative_delta_pct": round(self.relative_delta_pct, 3),
            "relative_reduction_pct": round(self.relative_reduction_pct, 3),
            "is_improvement": self.is_improvement,
        }


def _normalize_level(raw: Union[BaselineLevel, int, str]) -> BaselineLevel:
    if isinstance(raw, BaselineLevel):
        return raw
    if isinstance(raw, int):
        return BaselineLevel(raw)
    raw_str = str(raw).strip().lower()
    mapping = {
        "0": BaselineLevel.LEVEL_0_STATIC_MWPM,
        "level_0": BaselineLevel.LEVEL_0_STATIC_MWPM,
        "level_0_static_mwpm": BaselineLevel.LEVEL_0_STATIC_MWPM,
        "static_mwpm": BaselineLevel.LEVEL_0_STATIC_MWPM,
        "b0": BaselineLevel.LEVEL_0_STATIC_MWPM,
        "b0_factory_mwpm": BaselineLevel.LEVEL_0_STATIC_MWPM,
        "1": BaselineLevel.LEVEL_1_PERIODIC_SPITZ,
        "level_1": BaselineLevel.LEVEL_1_PERIODIC_SPITZ,
        "level_1_periodic_spitz": BaselineLevel.LEVEL_1_PERIODIC_SPITZ,
        "periodic_spitz": BaselineLevel.LEVEL_1_PERIODIC_SPITZ,
        "periodic_spitz_reweighting": BaselineLevel.LEVEL_1_PERIODIC_SPITZ,
        "b3": BaselineLevel.LEVEL_1_PERIODIC_SPITZ,
        "b3_online_reweighted_mwpm": BaselineLevel.LEVEL_1_PERIODIC_SPITZ,
        "2": BaselineLevel.LEVEL_2_CUSUM_GATED,
        "level_2": BaselineLevel.LEVEL_2_CUSUM_GATED,
        "level_2_cusum_gated": BaselineLevel.LEVEL_2_CUSUM_GATED,
        "cusum_gated": BaselineLevel.LEVEL_2_CUSUM_GATED,
        "cusum": BaselineLevel.LEVEL_2_CUSUM_GATED,
        "b4": BaselineLevel.LEVEL_2_CUSUM_GATED,
        "b4_naive_adaptive": BaselineLevel.LEVEL_2_CUSUM_GATED,
        "3": BaselineLevel.LEVEL_3_FULL_CG_CCADS,
        "level_3": BaselineLevel.LEVEL_3_FULL_CG_CCADS,
        "level_3_full_cg_ccads": BaselineLevel.LEVEL_3_FULL_CG_CCADS,
        "full_cg_ccads": BaselineLevel.LEVEL_3_FULL_CG_CCADS,
        "cg_ccads": BaselineLevel.LEVEL_3_FULL_CG_CCADS,
        "b5": BaselineLevel.LEVEL_3_FULL_CG_CCADS,
        "b5_selective_cg_ccads": BaselineLevel.LEVEL_3_FULL_CG_CCADS,
        "4": BaselineLevel.ORACLE_CEILING,
        "level_4": BaselineLevel.ORACLE_CEILING,
        "oracle": BaselineLevel.ORACLE_CEILING,
        "oracle_ceiling": BaselineLevel.ORACLE_CEILING,
        "oracle_mwpm": BaselineLevel.ORACLE_CEILING,
        "b6": BaselineLevel.ORACLE_CEILING,
        "b6_oracle": BaselineLevel.ORACLE_CEILING,
    }
    if raw_str in mapping:
        return mapping[raw_str]
    raise ValueError(f"Unknown baseline level identifier: {raw}")


def _extract_ler(val: Union[float, dict[str, Any]]) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        if "mean_ler" in val:
            return float(val["mean_ler"])
        if "ler" in val:
            return float(val["ler"])
        if "logical_error_rate" in val:
            return float(val["logical_error_rate"])
    raise ValueError(f"Cannot extract logical error rate from: {val}")


def compute_baseline_deltas(
    results: dict[Union[BaselineLevel, int, str], Union[float, dict[str, Any]]],
    reference_level: Union[BaselineLevel, int, str] = BaselineLevel.LEVEL_0_STATIC_MWPM,
) -> dict[str, BaselineDeltaBenchmark]:
    """
    Compute relative performance delta benchmarks of all baseline levels against Level 0 (Static MWPM).

    Negative relative_delta_pct indicates improved logical error suppression compared to Level 0.
    Positive relative_reduction_pct indicates percentage error reduction.

    Parameters
    ----------
    results : dict
        Mapping from BaselineLevel, int (0..4), or level key to LER or result dict.
    reference_level : BaselineLevel, int, or str
        Reference level to benchmark against (default: Level 0 Static MWPM).

    Returns
    -------
    dict[str, BaselineDeltaBenchmark]
        Keyed by level string (e.g. 'level_0_static_mwpm', 'level_1_periodic_spitz', ...).
    """
    ref_norm = _normalize_level(reference_level)
    ref_val = None

    # Find reference LER
    for k, v in results.items():
        if _normalize_level(k) == ref_norm:
            ref_val = _extract_ler(v)
            break

    if ref_val is None:
        raise ValueError(f"Reference level {ref_norm.name} not found in provided results")

    ref_denom = max(ref_val, 1e-12)
    benchmarks: dict[str, BaselineDeltaBenchmark] = {}

    for k, v in results.items():
        level = _normalize_level(k)
        ler = _extract_ler(v)
        info = BASELINE_LEVEL_REGISTRY[level]
        delta = ler - ref_val
        rel_delta_pct = (delta / ref_denom) * 100.0
        rel_reduction_pct = ((ref_val - ler) / ref_denom) * 100.0

        key = level.name.lower()
        benchmarks[key] = BaselineDeltaBenchmark(
            level=level,
            formal_name=info.formal_name,
            ler=ler,
            reference_ler=ref_val,
            delta_ler=delta,
            relative_delta_pct=rel_delta_pct,
            relative_reduction_pct=rel_reduction_pct,
            is_improvement=(ler < ref_val),
        )

    return benchmarks


# ---------------------------------------------------------------------------
# Baseline Arm and Ladder Manager
# ---------------------------------------------------------------------------

@dataclass
class BaselineArm:
    """Metadata and configured decoder instance for a baseline ladder rung."""
    rung: BaselineRung
    decoder: Decoder
    description: str
    is_oracle: bool = False
    is_adaptive: bool = False
    level: Optional[BaselineLevel] = None


class BaselineLadder:
    """
    Standard evaluation hierarchy for comparing adaptive and static QEC strategies.
    Spans the formal 5-level baseline hierarchy (Levels 0-4) as well as legacy rungs (B0-B6).
    """

    def __init__(self, circuit: stim.Circuit) -> None:
        self.circuit = circuit
        self.arms: dict[BaselineRung, BaselineArm] = {}
        self._initialize_ladder()

    def _initialize_ladder(self) -> None:
        """Instantiate and configure all ladder decoders."""
        # Level 0 / B0: Factory MWPM (default circuit DEM)
        b0_dec = MWPMDecoder()
        b0_dec.configure(circuit=self.circuit)
        self.arms[BaselineRung.B0_FACTORY_MWPM] = BaselineArm(
            rung=BaselineRung.B0_FACTORY_MWPM,
            decoder=b0_dec,
            description="Static factory MWPM using nominal circuit DEM weights",
            level=BaselineLevel.LEVEL_0_STATIC_MWPM,
        )

        # B1: Optimized Static MWPM (calibrated on baseline parameters)
        b1_calibrator = DEMCalibrator(circuit=self.circuit)
        calibrated_b1 = b1_calibrator.calibrate(measured_p_2q=0.003, measured_p_ro=0.01)
        b1_dec = MWPMDecoder()
        b1_dec.configure(matching=calibrated_b1.to_matching(), num_detectors=self.circuit.num_detectors)
        self.arms[BaselineRung.B1_OPTIMIZED_MWPM] = BaselineArm(
            rung=BaselineRung.B1_OPTIMIZED_MWPM,
            decoder=b1_dec,
            description="Optimized static MWPM with fixed calibrated baseline weights",
        )

        # B2: Genuine Cluster-Growth Union-Find
        b2_dec = UnionFindDecoder()
        b2_dec.configure(circuit=self.circuit)
        self.arms[BaselineRung.B2_TRUE_UNION_FIND] = BaselineArm(
            rung=BaselineRung.B2_TRUE_UNION_FIND,
            decoder=b2_dec,
            description="True cluster-growth Union-Find decoder (Delfosse & Nickerson)",
        )

        # Level 1 / B3: Online Reweighted MWPM (Bhardwaj et al. / Spitz sliding-window)
        b3_dec = OnlineReweightedMWPM(window_size=150, update_interval=30)
        b3_dec.configure(circuit=self.circuit)
        self.arms[BaselineRung.B3_ONLINE_REWEIGHTED] = BaselineArm(
            rung=BaselineRung.B3_ONLINE_REWEIGHTED,
            decoder=b3_dec,
            description="Online reweighted MWPM via sliding-window syndrome statistics (Bhardwaj et al.)",
            is_adaptive=True,
            level=BaselineLevel.LEVEL_1_PERIODIC_SPITZ,
        )

    def register_adaptive_arm(
        self,
        rung: BaselineRung,
        decoder: Decoder,
        description: str,
        is_oracle: bool = False,
        level: Optional[BaselineLevel] = None,
    ) -> None:
        """Register dynamic arms like B4 (naive), B5 (CG-CCADS), and B6 (oracle)."""
        self.arms[rung] = BaselineArm(
            rung=rung,
            decoder=decoder,
            description=description,
            is_oracle=is_oracle,
            is_adaptive=True,
            level=level,
        )

    def register_level(
        self,
        level: BaselineLevel,
        decoder: Decoder,
        description: Optional[str] = None,
    ) -> None:
        """Register a decoder arm by its formal 5-level baseline hierarchy level."""
        info = BASELINE_LEVEL_REGISTRY[level]
        rung = info.rung_equivalent or BaselineRung.B0_FACTORY_MWPM
        self.arms[rung] = BaselineArm(
            rung=rung,
            decoder=decoder,
            description=description or info.description,
            is_oracle=info.is_oracle,
            is_adaptive=info.is_adaptive,
            level=level,
        )

    def get_level_arm(self, level: BaselineLevel) -> Optional[BaselineArm]:
        """Retrieve the configured arm for a formal baseline level."""
        info = BASELINE_LEVEL_REGISTRY.get(level)
        if info and info.rung_equivalent in self.arms:
            return self.arms[info.rung_equivalent]
        for arm in self.arms.values():
            if arm.level == level:
                return arm
        return None

    def compare_deltas(
        self,
        results: dict[Union[BaselineLevel, int, str], Union[float, dict[str, Any]]],
        reference_level: Union[BaselineLevel, int, str] = BaselineLevel.LEVEL_0_STATIC_MWPM,
    ) -> dict[str, BaselineDeltaBenchmark]:
        """Compute relative performance delta benchmarks across ladder levels."""
        return compute_baseline_deltas(results, reference_level=reference_level)

    def to_counterfactual_engine(self) -> CounterfactualEngine:
        """Create a CounterfactualEngine wired to all ladder arms."""
        engine = CounterfactualEngine(reference_arm=BaselineRung.B0_FACTORY_MWPM.value)
        for rung, arm in self.arms.items():
            engine.register_decoder(rung.value, arm.decoder)
        return engine
