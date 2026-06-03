"""
QEC package: circuit construction, code definitions, and scheduling.

Modules:
    codes: QEC code definitions (surface, repetition, color, QLDPC)
    circuits: Stim circuit generation and Qiskit conversion
    schedules: Stabilizer measurement schedule definitions
    adaptive_scheduler: Dynamic X/Z stabilizer frequency adaptation
"""

from adaptive_qec.qec.schedules import (
    ScheduleType,
    StabilizerSchedule,
    get_schedule,
    create_custom_schedule,
    schedule_for_bias,
)
from adaptive_qec.qec.adaptive_scheduler import (
    AdaptiveSchedulerConfig,
    AdaptiveXZScheduler,
    DefectObservation,
)

__all__ = [
    "ScheduleType",
    "StabilizerSchedule",
    "get_schedule",
    "create_custom_schedule",
    "schedule_for_bias",
    "AdaptiveSchedulerConfig",
    "AdaptiveXZScheduler",
    "DefectObservation",
]
