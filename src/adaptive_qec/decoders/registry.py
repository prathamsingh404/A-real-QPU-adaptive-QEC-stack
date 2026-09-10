"""
Decoder plugin registry.

Register and instantiate decoders by name. Add new decoders without
modifying core code.
"""

from __future__ import annotations

import logging
from typing import Type

from adaptive_qec.decoders.base import Decoder
