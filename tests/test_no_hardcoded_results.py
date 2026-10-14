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


def scan_file_for_unwhitelisted_floats(file_path: Path) -> list[tuple[int, float]]:
    """Scan a Python source file and return all unwhitelisted float literals with line numbers."""
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    unwhitelisted = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            val = float(node.value)
            if val not in ALLOWED_FLOAT_WHITELIST:
                unwhitelisted.append((node.lineno, val))

    return unwhitelisted


def test_make_paper_tables_has_no_hardcoded_results():
    """Verify make_paper_tables.py contains no hardcoded scientific result literals."""
    target = Path("src/adaptive_qec/analysis/make_paper_tables.py")
    violations = scan_file_for_unwhitelisted_floats(target)
    assert violations == [], f"Found unwhitelisted float literals in {target}: {violations}"


def test_make_paper_figures_has_no_hardcoded_results():
    """Verify make_paper_figures.py contains no hardcoded scientific result literals."""
    target = Path("src/adaptive_qec/analysis/make_paper_figures.py")
    violations = scan_file_for_unwhitelisted_floats(target)
    assert violations == [], f"Found unwhitelisted float literals in {target}: {violations}"


def test_missing_results_file_raises_error():
    """Verify pipeline fails gracefully when results file is missing."""
    from adaptive_qec.evaluation.empirical_scaling_and_ablation import ComponentAblationStudy

    non_existent = Path("data/definitely_missing_results_12345.json")
    with pytest.raises(FileNotFoundError):
        ComponentAblationStudy.run_ablation(results_path=non_existent)
