"""
Decoder plugin registry.

Register and instantiate decoders by name. Add new decoders without
modifying core code.
"""

from __future__ import annotations

import logging
from typing import Type

from adaptive_qec.decoders.base import Decoder

logger = logging.getLogger(__name__)

_DECODERS: dict[str, Type[Decoder]] = {}


def register_decoder(name: str, decoder_class: Type[Decoder]) -> None:
    """Register a decoder class."""
    _DECODERS[name.lower()] = decoder_class
    logger.debug(f"Registered decoder: {name} → {decoder_class.__name__}")


def get_decoder(name: str) -> Decoder:
    """
    Instantiate a decoder by name.

    Built-in decoders are lazy-loaded.
    """
    name_lower = name.lower()

    # Lazy-register built-in decoders
    if name_lower == "mwpm" and name_lower not in _DECODERS:
        from adaptive_qec.decoders.mwpm import MWPMDecoder
        register_decoder("mwpm", MWPMDecoder)

    if name_lower == "union_find" and name_lower not in _DECODERS:
        from adaptive_qec.decoders.union_find import UnionFindDecoder
        register_decoder("union_find", UnionFindDecoder)

    if name_lower not in _DECODERS:
        available = list(_DECODERS.keys()) or ["none registered"]
        raise ValueError(
            f"Unknown decoder '{name}'. "
            f"Available: {', '.join(available)}. "
            f"Register new decoders with register_decoder()."
        )

    decoder = _DECODERS[name_lower]()
    logger.info(f"Created decoder: {decoder.name}")
    return decoder


def list_decoders() -> list[str]:
    """List all registered decoder names."""
    return list(_DECODERS.keys())
