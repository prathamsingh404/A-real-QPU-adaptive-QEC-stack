"""
Sequential Probability Ratio Test (SPRT) augmented controller.

Implements Wald's SPRT to make statistically rigorous switching
decisions between QEC modes.  Instead of switching whenever a cost
estimate looks better, we accumulate evidence until the log-likelihood
ratio crosses a threshold — guaranteeing bounded Type-I/II error.

Mathematical formulation:
    H₀: current mode's error rate ≤ challenger's error rate
    H₁: challenger's error rate < current mode's error rate − Δ

    Log-likelihood ratio:
        Λₙ = Σᵢ log[ P(xᵢ | H₁) / P(xᵢ | H₀) ]

    Decision boundaries:
        Λₙ ≥  log((1−β)/α)  →  accept H₁ (switch)
        Λₙ ≤  log(β/(1−α))  →  accept H₀ (stay)

    where α = P(Type-I error), β = P(Type-II error).

This module can wrap any BaseController to gate its switching
decisions through SPRT, or operate standalone as a SPRT-bandit hybrid.

References:
    - Wald, "Sequential Tests of Statistical Hypotheses" (1945)
    - Roadmap: PROJECT_EVOLUTION_ROADMAP.md §Phase-3
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from adaptive_qec.controller.base import BaseController
from adaptive_qec.controller.controller import (
    ControlAction,
    CostWeights,
    DecoderChoice,
    HardwareState,
)
from adaptive_qec.mitigation.dynamical_decoupling import DDSequenceType
