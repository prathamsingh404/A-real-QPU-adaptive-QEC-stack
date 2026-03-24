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
