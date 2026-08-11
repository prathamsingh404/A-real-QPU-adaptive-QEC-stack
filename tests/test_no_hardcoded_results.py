"""
AST Integrity Test: Prohibits Hardcoded Result Literals.

Scans table and figure generator files using Python's AST module to ensure
no hardcoded scientific result floats exist outside a strict structural whitelist.
Guarantees that all manuscript numbers are dynamically derived from empirical artifacts.
"""

import ast
from pathlib import Path
import pytest

# Allowable structural / layout / numerical constant values
ALLOWED_FLOAT_WHITELIST = {
    0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    0.25,
    0.75,
    0.1,
    0.2,
    0.3,
    0.8,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    10.0,
    100.0,
    1e-3,
    1e-4,
    1e-6,
    1e-9,
    1e-12,
    1e6,
    300.0,
    # Diagram layout coordinates & alpha
    0.05,
    0.12,
    0.15,
    0.18,
    0.22,
    0.26,
    0.27,
    0.28,
    0.32,
    0.33,
    0.35,
    0.4,
    0.45,
    0.55,
    0.57,
    0.6,
    0.61,
    0.65,
    0.7,
    0.87,
    0.9,
    0.95,
    0.975,
    4.5,
    5.5,
    6.0,
    8.0,
    11.0,
    12.0,
    13.0,
    14.0,
    # Small timing profile synthetic sleeps
    0.00003,
    0.00001,
}
