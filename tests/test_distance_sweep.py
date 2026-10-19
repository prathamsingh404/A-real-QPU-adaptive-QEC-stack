import pytest
from scripts.run_distance_sweep_adaptive import main
def test_distance_sweep_callable():
    assert callable(main)
