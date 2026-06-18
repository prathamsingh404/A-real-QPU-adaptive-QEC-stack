"""
The Complete 7-Rung Baseline Ladder for Adaptive QEC (TASK-018, Point 10).

Formally defines and instantiates the standard baseline hierarchy:
    B0: Static Factory MWPM (nominal circuit calibration weights)
    B1: Optimized Static MWPM (best fixed weights fitted to stationary baseline)
    B2: Genuine Cluster-Growth Union-Find (Delfosse & Nickerson)
    B3: Online Reweighted MWPM (syndrome sliding-window DGR / Bhardwaj et al. PRX Quantum 2026)
    B4: Naive Adaptive Policy (historical reactive controller, motivating negative baseline)
    B5: Selective Adaptive Controller (CG-CCADS, core novel contribution)
    B6: Oracle Policy (privileged access to true latent noise, theoretical upper bound)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import stim

from adaptive_qec.decoders.base import Decoder
from adaptive_qec.decoders.mwpm import MWPMDecoder
from adaptive_qec.decoders.reweighted_mwpm import OnlineReweightedMWPM
from adaptive_qec.decoders.union_find import UnionFindDecoder
from adaptive_qec.decoders.dem_calibration import DEMCalibrator
from adaptive_qec.evaluation.counterfactual import CounterfactualEngine

logger = logging.getLogger(__name__)


class BaselineRung(str, Enum):
    B0_FACTORY_MWPM = "b0_factory_mwpm"
    B1_OPTIMIZED_MWPM = "b1_optimized_mwpm"
    B2_TRUE_UNION_FIND = "b2_true_union_find"
    B3_ONLINE_REWEIGHTED = "b3_online_reweighted_mwpm"
    B4_NAIVE_ADAPTIVE = "b4_naive_adaptive"
    B5_SELECTIVE_CG_CCADS = "b5_selective_cg_ccads"
    B6_ORACLE = "b6_oracle"


@dataclass
class BaselineArm:
    """Metadata and configured decoder instance for a baseline ladder rung."""
    rung: BaselineRung
    decoder: Decoder
    description: str
    is_oracle: bool = False
    is_adaptive: bool = False


class BaselineLadder:
    """
    Standard evaluation hierarchy for comparing adaptive and static QEC strategies.
    """

    def __init__(self, circuit: stim.Circuit) -> None:
        self.circuit = circuit
        self.arms: dict[BaselineRung, BaselineArm] = {}
        self._initialize_ladder()

    def _initialize_ladder(self) -> None:
        """Instantiate and configure all ladder decoders."""
        # B0: Factory MWPM (default circuit DEM)
        b0_dec = MWPMDecoder()
        b0_dec.configure(circuit=self.circuit)
        self.arms[BaselineRung.B0_FACTORY_MWPM] = BaselineArm(
            rung=BaselineRung.B0_FACTORY_MWPM,
            decoder=b0_dec,
            description="Static factory MWPM using nominal circuit DEM weights",
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
