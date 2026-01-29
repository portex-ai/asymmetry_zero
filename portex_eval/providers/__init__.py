"""Model provider abstractions for portex-eval.

Providers wrap external LLM APIs and expose a uniform interface
for generation calls. Currently supports:

- OpenRouter: openrouter:<model_id> (e.g., openrouter:google/gemini-2.5-flash)

Usage:
    from portex_eval.providers import get_provider

    provider = get_provider("openrouter:google/gemini-2.5-flash")
    response = provider.generate("What is 2+2?")
    print(response.text)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from portex_eval.providers.base import Provider, Response, parse_model_string
from portex_eval.providers.openrouter import (
    OpenRouterProvider,
    RateLimitError,
    create_openrouter_provider,
)

__all__ = [
    "Provider",
    "Response",
    "OpenRouterProvider",
    "RateLimitError",
    "get_provider",
    "parse_model_string",
]

# Registry of provider factories
ProviderFactory = Callable[..., Provider]

_PROVIDER_REGISTRY: dict[str, ProviderFactory] = {
    "openrouter": create_openrouter_provider,
}


def get_provider(model_string: str, **kwargs: Any) -> Provider:
    """Get a provider instance from a model string.

    Args:
        model_string: A string in format 'provider:model_id'.
            Examples:
                - 'openrouter:google/gemini-2.5-flash'
                - 'openrouter:anthropic/claude-3.5-sonnet'
        **kwargs: Additional arguments passed to the provider constructor.

    Returns:
        An initialized Provider instance.

    Raises:
        ValueError: If the provider is not supported or model string is invalid.
    """
    provider_id, model_id = parse_model_string(model_string)

    if provider_id not in _PROVIDER_REGISTRY:
        supported = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
        raise ValueError(f"Unknown provider '{provider_id}'. Supported providers: {supported}")

    factory = _PROVIDER_REGISTRY[provider_id]
    return factory(model_id, **kwargs)


def register_provider(provider_id: str, factory: ProviderFactory) -> None:
    """Register a custom provider factory.

    Args:
        provider_id: The provider identifier (e.g., 'custom').
        factory: A callable that takes (model_id, **kwargs) and returns a Provider.
    """
    _PROVIDER_REGISTRY[provider_id] = factory
