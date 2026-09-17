"""Mitigation package for quantum error correction.

Includes Dynamical Decoupling (DD), crosstalk suppression, and calibration tuning.
"""

from adaptive_qec.mitigation.dynamical_decoupling import (
    AdaptiveDDPlanner,
    DDSchedule,
    DDSequenceType,
)

__all__ = [
