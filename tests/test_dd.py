"""Tests for Dynamical Decoupling Integration (Problem 3).

Validates selective DD candidate decision rule, sequence selection (CPMG, XY4, XY8),
idle time estimation from circuits, and Digital Twin DD candidate querying.
"""

import pytest
import stim

from adaptive_qec.config import NoiseConfig
from adaptive_qec.digital_twin.twin import HardwareDigitalTwin
from adaptive_qec.mitigation.dynamical_decoupling import (
    AdaptiveDDPlanner,
    DDSchedule,
    DDSequenceType,
)
from adaptive_qec.qec.codes import create_code


class TestDynamicalDecoupling:
    """Tests for AdaptiveDDPlanner and DDSchedule."""
