"""
QPU backend registry.

Plugin system for registering and instantiating QPU backends.
Supports IBM Quantum out of the box, with extension points for
IQM, Quantinuum, and other providers.
"""

from __future__ import annotations

import logging
from typing import Any, Type

from adaptive_qec.config import AdaptiveQECConfig, HardwareProvider
from adaptive_qec.qpu.base import QPUBackend

logger = logging.getLogger(__name__)

# Global registry of backend classes
_BACKENDS: dict[str, Any] = {}


def register_backend(provider: str, backend_class: Any) -> None:
    """Register a QPU backend class for a given provider name."""
    _BACKENDS[provider.lower()] = backend_class
    logger.debug(f"Registered QPU backend: {provider} → {backend_class.__name__}")


def get_backend(config: AdaptiveQECConfig) -> QPUBackend:
    """
    Instantiate the appropriate QPU backend from configuration.

    Args:
        config: Full AdaptiveQEC configuration.

    Returns:
        Instantiated QPUBackend.

    Raises:
        ValueError: If the provider is not registered.
    """
    provider = config.hardware.provider.value

    # Lazy-register built-in backends
    if provider == HardwareProvider.IBM.value and provider not in _BACKENDS:
        from adaptive_qec.qpu.ibm import IBMQuantumBackend
        register_backend(provider, IBMQuantumBackend)

    if provider not in _BACKENDS:
        available = list(_BACKENDS.keys()) or ["none registered"]
        raise ValueError(
            f"Unknown QPU provider '{provider}'. "
            f"Available: {', '.join(available)}. "
            f"Register new backends with register_backend()."
        )

    backend_class = _BACKENDS[provider]
    backend = backend_class(config.hardware)
    logger.info(f"Created QPU backend: {backend_class.__name__} for provider '{provider}'")
    return backend


def list_backends() -> list[str]:
    """List all registered backend provider names."""
    return list(_BACKENDS.keys())
