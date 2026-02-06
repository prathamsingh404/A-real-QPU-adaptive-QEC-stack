"""
Filesystem-based experiment data store.

Implements the experiment storage layout from the spec:

    experiments/{experiment_id}/
        config.yaml
        circuit.qasm
        qpu_metadata.json
        calibration.json
        raw_results/
            measurement_outcomes.npy
        detector_data/
            syndrome_tensor.npy
            observable_flips.npy
            detector_metadata.json
        decoder_results/
            {decoder_name}_corrections.npy
            {decoder_name}_results.json
        metrics.json
        plots/
        git_commit.txt

Six months later: `reproduce experiment_0421` should work.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from adaptive_qec.data.models import (
    DecoderRecord,
    DetectorRecord,
    ExperimentMetrics,
    ExperimentRecord,
)

logger = logging.getLogger(__name__)


class ExperimentStore:
    """
    Filesystem-based experiment data store.

    Every experiment creates a self-contained directory with all data
    needed for reproduction.
    """

    def __init__(self, base_path: str | Path) -> None:
        self._base = Path(base_path)
        self._base.mkdir(parents=True, exist_ok=True)

    def experiment_path(self, experiment_id: str) -> Path:
        """Get the directory path for an experiment."""
        return self._base / experiment_id

    def save_experiment(self, record: ExperimentRecord) -> Path:
        """
        Save a complete experiment record.

        Creates the full directory structure with all artifacts.
        """
        exp_dir = self.experiment_path(record.experiment_id)
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        raw_dir = exp_dir / "raw_results"
        raw_dir.mkdir(exist_ok=True)
        detector_dir = exp_dir / "detector_data"
        detector_dir.mkdir(exist_ok=True)
        decoder_dir = exp_dir / "decoder_results"
        decoder_dir.mkdir(exist_ok=True)
        plots_dir = exp_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
