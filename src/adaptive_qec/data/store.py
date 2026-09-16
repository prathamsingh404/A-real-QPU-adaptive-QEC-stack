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

        # Save config snapshot
        with open(exp_dir / "config.yaml", "w") as f:
            yaml.dump(record.config_snapshot, f, default_flow_style=False)

        # Save circuit QASM
        with open(exp_dir / "circuit.qasm", "w") as f:
            f.write(record.circuit_qasm)

        # Save QPU metadata (everything except raw measurements)
        with open(exp_dir / "qpu_metadata.json", "w") as f:
            json.dump(record.to_dict(), f, indent=2, default=str)

        # Save calibration snapshot
        with open(exp_dir / "calibration.json", "w") as f:
            json.dump(record.calibration_snapshot, f, indent=2, default=str)

        # Save raw measurement outcomes as numpy array
        if record.measurement_outcomes.size > 0:
            np.save(raw_dir / "measurement_outcomes.npy", record.measurement_outcomes)

        # Save git commit
        if record.git_commit:
            with open(exp_dir / "git_commit.txt", "w") as f:
                f.write(record.git_commit)

        logger.info(f"Saved experiment {record.experiment_id} to {exp_dir}")
        return exp_dir

    def save_detector_data(self, record: DetectorRecord) -> None:
        """Save detector extraction results."""
        detector_dir = self.experiment_path(record.experiment_id) / "detector_data"
        detector_dir.mkdir(parents=True, exist_ok=True)

        np.save(detector_dir / "syndrome_tensor.npy", record.syndrome_tensor)
        np.save(detector_dir / "observable_flips.npy", record.observable_flips)

        if record.detector_coordinates is not None:
            np.save(detector_dir / "detector_coordinates.npy", record.detector_coordinates)

        with open(detector_dir / "detector_metadata.json", "w") as f:
            json.dump(record.to_dict(), f, indent=2)

        logger.info(
            f"Saved detector data for {record.experiment_id}: "
            f"{record.num_detectors} detectors, {record.num_rounds} rounds"
        )

    def save_decoder_results(self, record: DecoderRecord) -> None:
        """Save decoder results."""
        decoder_dir = self.experiment_path(record.experiment_id) / "decoder_results"
        decoder_dir.mkdir(parents=True, exist_ok=True)

        np.save(
            decoder_dir / f"{record.decoder_name}_corrections.npy",
            record.corrections,
        )

        if record.per_shot_latency_us is not None:
            np.save(
                decoder_dir / f"{record.decoder_name}_latency.npy",
                record.per_shot_latency_us,
            )

        with open(decoder_dir / f"{record.decoder_name}_results.json", "w") as f:
            json.dump(record.to_dict(), f, indent=2, default=str)

        logger.info(
            f"Saved decoder results for {record.experiment_id}: "
            f"{record.decoder_name} — LER={record.logical_error_rate:.6f}"
        )

    def save_metrics(self, metrics: ExperimentMetrics) -> None:
        """Save experiment metrics."""
        exp_dir = self.experiment_path(metrics.experiment_id)
        exp_dir.mkdir(parents=True, exist_ok=True)

        with open(exp_dir / "metrics.json", "w") as f:
            json.dump(metrics.to_dict(), f, indent=2, default=str)

        logger.info(f"Saved metrics for {metrics.experiment_id}")

    def save_plot(self, experiment_id: str, name: str, fig: Any) -> Path:
        """Save a matplotlib figure to the plots directory."""
        plots_dir = self.experiment_path(experiment_id) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        path = plots_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot: {path}")
        return path

    def load_experiment(self, experiment_id: str) -> ExperimentRecord:
        """Load an experiment record from disk."""
        exp_dir = self.experiment_path(experiment_id)
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment not found: {experiment_id}")

        with open(exp_dir / "qpu_metadata.json") as f:
            data = json.load(f)

        record = ExperimentRecord.from_dict(data)

        # Load raw measurements if available
        raw_path = exp_dir / "raw_results" / "measurement_outcomes.npy"
        if raw_path.exists():
            record.measurement_outcomes = np.load(raw_path)

        return record

    def load_detector_data(self, experiment_id: str) -> DetectorRecord:
        """Load detector data from disk."""
        detector_dir = self.experiment_path(experiment_id) / "detector_data"

        with open(detector_dir / "detector_metadata.json") as f:
            meta = json.load(f)

        syndrome = np.load(detector_dir / "syndrome_tensor.npy")
        obs = np.load(detector_dir / "observable_flips.npy")
        coords = None
        coords_path = detector_dir / "detector_coordinates.npy"
        if coords_path.exists():
            coords = np.load(coords_path)

        return DetectorRecord(
            experiment_id=meta["experiment_id"],
            num_rounds=meta["num_rounds"],
            num_detectors=meta["num_detectors"],
            syndrome_tensor=syndrome,
            observable_flips=obs,
            detector_coordinates=coords,
        )

    def load_metrics(self, experiment_id: str) -> dict[str, Any]:
        """Load experiment metrics from disk."""
        metrics_path = self.experiment_path(experiment_id) / "metrics.json"
        with open(metrics_path) as f:
            return json.load(f)

    def list_experiments(self) -> list[str]:
        """List all experiment IDs in the store."""
        experiments = []
        for path in sorted(self._base.iterdir()):
            if path.is_dir() and (path / "qpu_metadata.json").exists():
                experiments.append(path.name)
        return experiments

    def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment and all its data."""
        exp_dir = self.experiment_path(experiment_id)
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
            logger.info(f"Deleted experiment: {experiment_id}")
