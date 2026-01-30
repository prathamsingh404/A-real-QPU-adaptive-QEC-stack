"""
Adaptive QEC controller package.

Provides the full suite of controllers for the adaptive QEC stack:
    - BaseController: Abstract interface all controllers implement
    - StaticController: Fixed-policy baseline (no adaptation)
    - CostBasedController: Heuristic switching with cost function
    - Exp3Controller: Adversarial bandit (exponential weights)
    - Exp3PController: Exp3 with high-probability bounds
    - DASEController: Drift-Aware Successive Elimination
    - SPRTController: Sequential Probability Ratio Test gated bandit

Factory function:
    create_controller(name, **kwargs) → BaseController
"""

from adaptive_qec.controller.base import (
    BaseController,
    ControlAction,
    HardwareState,
    TelemetryRecord,
)
from adaptive_qec.controller.static import StaticController
from adaptive_qec.controller.cost_based import CostBasedController
from adaptive_qec.controller.bandit import (
    BanditArm,
    DASEController,
    Exp3Controller,
    Exp3PController,
    build_arm_set,
)
