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
