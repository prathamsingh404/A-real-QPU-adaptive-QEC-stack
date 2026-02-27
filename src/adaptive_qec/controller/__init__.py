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
from adaptive_qec.controller.sprt import (
    SPRTController,
    SPRTEngine,
    SPRTState,
)


CONTROLLER_REGISTRY: dict[str, type[BaseController]] = {
    "static": StaticController,
    "cost_based": CostBasedController,
    "exp3": Exp3Controller,
    "exp3p": Exp3PController,
    "dase": DASEController,
    "sprt": SPRTController,
}


def create_controller(name: str, **kwargs) -> BaseController:
    """
    Factory function to create a controller by name.

    Args:
        name: Controller type name (see CONTROLLER_REGISTRY).
        **kwargs: Controller-specific configuration.

    Returns:
        Instantiated controller.

    Raises:
        ValueError: If name is not recognized.
    """
    if name not in CONTROLLER_REGISTRY:
        available = ", ".join(sorted(CONTROLLER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown controller '{name}'. Available: {available}"
        )
    return CONTROLLER_REGISTRY[name](**kwargs)


__all__ = [
    "BaseController",
    "ControlAction",
    "HardwareState",
    "TelemetryRecord",
    "StaticController",
    "CostBasedController",
    "Exp3Controller",
    "Exp3PController",
    "DASEController",
    "SPRTController",
    "SPRTEngine",
    "SPRTState",
    "BanditArm",
    "build_arm_set",
    "create_controller",
    "CONTROLLER_REGISTRY",
]
