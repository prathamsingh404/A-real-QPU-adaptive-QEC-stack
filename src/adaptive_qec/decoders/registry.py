"""
Decoder Registry with Lazy MWPM Hybrid Support
"""
from __future__ import annotations
from typing import Type
from adaptive_qec.decoders.base import Decoder
from adaptive_qec.decoders.mwpm import MWPMDecoder
from adaptive_qec.decoders.union_find import UnionFindDecoder
from adaptive_qec.decoders.lazy_mwpm import LazyMWPMDecoder

DECODER_REGISTRY: dict[str, Type[Decoder]] = {
    "mwpm": MWPMDecoder,
    "union_find": UnionFindDecoder,
    "uf": UnionFindDecoder,
    "lazy_mwpm": LazyMWPMDecoder,
}

def get_decoder(name: str, **kwargs) -> Decoder:
    name_clean = name.lower().strip()
    if name_clean not in DECODER_REGISTRY:
        raise ValueError(f"Unknown decoder: {name}. Available: {list(DECODER_REGISTRY.keys())}")
    return DECODER_REGISTRY[name_clean](**kwargs)

def list_decoders() -> list[str]:
    return sorted(list(DECODER_REGISTRY.keys()))
