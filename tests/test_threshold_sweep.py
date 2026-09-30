import pytest
from scripts.run_threshold_sweep import main
def test_threshold_sweep_callable():
    assert callable(main)
