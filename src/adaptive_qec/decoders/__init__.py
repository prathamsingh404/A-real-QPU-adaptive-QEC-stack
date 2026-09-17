"""Decoder framework — plugin-based decoder architecture."""

from adaptive_qec.decoders.base import Correction, Decoder, DecoderMetrics
from adaptive_qec.decoders.mwpm import MWPMDecoder
from adaptive_qec.decoders.union_find import UnionFindDecoder, UnionFindForest
from adaptive_qec.decoders.registry import get_decoder, list_decoders, register_decoder

__all__ = [
    "Correction",
    "Decoder",
    "DecoderMetrics",
    "MWPMDecoder",
