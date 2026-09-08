"""
Counterfactual Decoding and Paired Evaluation Engine.

Provides mathematically rigorous paired evaluation of decoders and adaptive policies
on identical syndrome bitstreams (TASK-020, TASK-021, Point 28, Point 29).

Key Properties:
    1. Feeds the exact same syndrome stream s_{1:N} to all classical decoders.
    2. Completely eliminates shot-by-shot syndrome variance between arms.
    3. Computes pairwise decoder disagreement C_t (essential for Gate 2 of CG-CCADS).
    4. Computes paired deltas Delta_i = e_{i, adaptive} - e_{i, static} and
       paired statistical significance tests (paired t-test and Wilcoxon signed-rank).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from scipy import stats

from adaptive_qec.decoders.base import Decoder, DecoderMetrics

logger = logging.getLogger(__name__)


def wilson_score_interval(errors: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score interval for binomial proportions."""
    if total == 0:
        return 0.0, 0.0
    p = errors / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denom
    spread = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return float(max(0.0, centre - spread)), float(min(1.0, centre + spread))


@dataclass
class PairedEvaluationResult:
    """Rigorous paired evaluation metrics for a single decoder or policy arm."""
    arm_name: str
    total_shots: int
    num_logical_errors: int
    logical_error_rate: float
    ler_wilson_ci: tuple[float, float]
    predictions: np.ndarray  # shape (shots, observables)
    error_mask: np.ndarray   # shape (shots,), True where prediction != actual

    # Paired metrics relative to reference baseline (typically Factory MWPM B0)
    delta_errors: Optional[np.ndarray] = None  # +1 if worse than B0, -1 if better, 0 if identical
    mean_delta_ler: float = 0.0                # LER(arm) - LER(B0)
    delta_std_err: float = 0.0
    p_value_paired_t: float = 1.0
    p_value_wilcoxon: float = 1.0
    decode_time_s: float = 0.0
    latency_mean_us: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_name": self.arm_name,
            "total_shots": self.total_shots,
            "num_logical_errors": self.num_logical_errors,
            "logical_error_rate": self.logical_error_rate,
            "ler_wilson_ci": list(self.ler_wilson_ci),
            "mean_delta_ler": self.mean_delta_ler,
            "delta_std_err": self.delta_std_err,
            "p_value_paired_t": self.p_value_paired_t,
            "p_value_wilcoxon": self.p_value_wilcoxon,
            "decode_time_s": self.decode_time_s,
            "latency_mean_us": self.latency_mean_us,
        }


@dataclass
class DisagreementMatrix:
    """Pairwise decoder disagreement rates C_t on identical syndromes."""
    decoder_names: list[str]
    matrix: np.ndarray  # shape (num_decoders, num_decoders), entries in [0, 1]

    def get(self, name_a: str, name_b: str) -> float:
        idx_a = self.decoder_names.index(name_a)
        idx_b = self.decoder_names.index(name_b)
        return float(self.matrix[idx_a, idx_b])

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for i, a in enumerate(self.decoder_names):
            out[a] = {}
            for j, b in enumerate(self.decoder_names):
                out[a][b] = float(self.matrix[i, j])
        return out


class CounterfactualEngine:
    """
    Executes counterfactual evaluation of multiple decoders on identical syndrome bitstreams.
    """

    def __init__(self, reference_arm: str = "b0_factory_mwpm") -> None:
        self.reference_arm = reference_arm
        self._decoders: dict[str, Decoder] = {}

    def register_decoder(self, name: str, decoder: Decoder) -> None:
        """Register a decoder arm under a descriptive baseline name."""
        self._decoders[name] = decoder

    def evaluate(
        self,
        syndromes: np.ndarray,
        observable_flips: np.ndarray,
    ) -> tuple[dict[str, PairedEvaluationResult], DisagreementMatrix]:
        """
        Replay syndromes through all registered decoders and compute paired comparisons.

        Args:
            syndromes: shape (shots, num_detectors), uint8
            observable_flips: shape (shots, num_observables), uint8

        Returns:
            Tuple of (dict of PairedEvaluationResult, DisagreementMatrix)
        """
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        observable_flips = np.asarray(observable_flips, dtype=np.uint8)
        shots = syndromes.shape[0]

        raw_results: dict[str, Any] = {}

        # 1. Decode through all arms
        for name, dec in self._decoders.items():
            metrics: DecoderMetrics = dec.decode_batch(syndromes, observable_flips)
            corr = dec.decode(syndromes)
            preds = np.asarray(corr.observable_corrections, dtype=np.uint8)
            if preds.ndim == 1:
                preds = preds.reshape(shots, -1)
            actual = observable_flips.reshape(shots, -1)
            err_mask = np.any(preds != actual, axis=1)

            ci = wilson_score_interval(int(np.sum(err_mask)), shots)
            raw_results[name] = {
                "metrics": metrics,
                "preds": preds,
                "err_mask": err_mask,
                "ci": ci,
            }

        # 2. Extract reference arm error mask
        ref_mask = None
        if self.reference_arm in raw_results:
            ref_mask = raw_results[self.reference_arm]["err_mask"]
        elif raw_results:
            first_arm = next(iter(raw_results.keys()))
            ref_mask = raw_results[first_arm]["err_mask"]

        # 3. Compute paired deltas and significance
        paired_results: dict[str, PairedEvaluationResult] = {}
        for name, data in raw_results.items():
            err_mask = data["err_mask"]
            num_err = int(np.sum(err_mask))
            ler = float(num_err / max(shots, 1))

            mean_delta = 0.0
            delta_se = 0.0
            p_t = 1.0
            p_wilcoxon = 1.0
            delta_arr = None

            if ref_mask is not None:
                # delta_i: +1 if arm erred and ref didn't; -1 if ref erred and arm didn't; 0 if agreed
                delta_arr = err_mask.astype(np.int32) - ref_mask.astype(np.int32)
                mean_delta = float(np.mean(delta_arr))
                if shots > 1:
                    delta_se = float(np.std(delta_arr, ddof=1) / np.sqrt(shots))
                    if np.any(delta_arr != 0):
                        t_stat, p_t = stats.ttest_1samp(delta_arr, 0.0)
                        p_t = float(p_t)
                        try:
                            # Non-zero differences for Wilcoxon
                            diffs = delta_arr[delta_arr != 0]
                            if len(diffs) >= 10:
                                _, p_wilcoxon = stats.wilcoxon(diffs)
                                p_wilcoxon = float(p_wilcoxon)
                        except Exception:
                            p_wilcoxon = 1.0

            paired_results[name] = PairedEvaluationResult(
                arm_name=name,
                total_shots=shots,
                num_logical_errors=num_err,
                logical_error_rate=ler,
                ler_wilson_ci=data["ci"],
                predictions=data["preds"],
                error_mask=err_mask,
                delta_errors=delta_arr,
                mean_delta_ler=mean_delta,
                delta_std_err=delta_se,
                p_value_paired_t=p_t,
                p_value_wilcoxon=p_wilcoxon,
                decode_time_s=data["metrics"].decode_time_s,
                latency_mean_us=data["metrics"].latency_mean_us,
            )

        # 4. Compute pairwise disagreement matrix
        arm_names = list(self._decoders.keys())
        k = len(arm_names)
        matrix = np.zeros((k, k), dtype=np.float64)

        for i, a in enumerate(arm_names):
            preds_a = raw_results[a]["preds"]
            for j, b in enumerate(arm_names):
                if i == j:
                    matrix[i, j] = 0.0
                else:
                    preds_b = raw_results[b]["preds"]
                    # Fraction of shots with differing observable corrections
                    disagreements = np.any(preds_a != preds_b, axis=1)
                    matrix[i, j] = float(np.mean(disagreements))

        return paired_results, DisagreementMatrix(decoder_names=arm_names, matrix=matrix)
