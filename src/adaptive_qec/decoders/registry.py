"""
Decoder Registry with Lazy MWPM Hybrid Support
"""
from __future__ import annotations
from typing import Type
from adaptive_qec.decoders.base import Decoder
from adaptive_qec.decoders.mwpm import MWPMDecoder
from adaptive_qec.decoders.union_find import UnionFindDecoder
from adaptive_qec.decoders.lazy_mwpm import LazyMWPMDecoder
from adaptive_qec.decoders.reweighted_mwpm import OnlineReweightedMWPM
