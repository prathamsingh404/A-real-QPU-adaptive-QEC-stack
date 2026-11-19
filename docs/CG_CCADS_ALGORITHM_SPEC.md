# Controller specification

The historical heuristic remains a comparator. The evaluated selective controller compares each completed window's detector counts to an independent reference, adjusts for all planned detector tests and looks, and rebuilds with a 50:50 empirical/factory mixture on alarm entry. Clearing the alarm restores the cached factory graph. Each trajectory allows two rebuilds. Decisions take effect on the next window. [Protocol](../PREREG.md) · [Executable policy](../src/adaptive_qec/experiments/causal_reweighting.py)
