"""
Lazy MWPM Hybrid Decoder Architecture
"""
from __future__ import annotations
import logging
import time
from typing import Any
import numpy as np
from adaptive_qec.decoders.base import Correction, Decoder, DecoderMetrics
from adaptive_qec.decoders.mwpm import MWPMDecoder
from adaptive_qec.decoders.union_find import UnionFindDecoder

logger = logging.getLogger(__name__)
