"""Model provider abstractions for portex-eval."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from portex_eval.providers.anthropic import AnthropicProvider, create_anthropic_provider
from portex_eval.providers.base import (
    ModelConfig,
    ModelSpec,
    Provider,
    Response,
    model_config_from_spec,
    model_config_to_dict,
    parse_model_string,
)
from portex_eval.providers.openai import OpenAIProvider, create_openai_provider
from portex_eval.providers.openai_compatible import (
    OpenAICompatibleProvider,
    OpenAICompatibleRateLimitError,
    create_openai_compatible_provider,
)
from portex_eval.providers.openrouter import (
    OpenRouterProvider,
    RateLimitError,
    create_openrouter_provider,
)

__all__ = [
    "Provider",
    "Response",
    "ModelConfig",
    "ModelSpec",
    "OpenRouterProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OpenAICompatibleProvider",
    "RateLimitError",
    "OpenAICompatibleRateLimitError",
    "get_provider",
    "get_supported_providers",
    "model_config_from_spec",
    "model_config_to_dict",
    "parse_model_string",
]

# Registry of provider factories
ProviderFactory = Callable[..., Provider]

_PROVIDER_REGISTRY: dict[str, ProviderFactory] = {
    "openrouter": create_openrouter_provider,
    "openai": create_openai_provider,
    "anthropic": create_anthropic_provider,
    "openai_compatible": create_openai_compatible_provider,
    "openai-compatible": create_openai_compatible_provider,
    "vllm": create_openai_compatible_provider,
    "custom": create_openai_compatible_provider,
}


def get_provider(model_spec: ModelSpec, **kwargs: Any) -> Provider:
    """Get a provider instance from a model string or config object.

    Args:
        model_spec: A string in format ``provider:model`` or a config object.
        **kwargs: Additional arguments passed to the provider constructor.

    Returns:
        An initialized Provider instance.

    Raises:
        ValueError: If the provider is not supported or model string is invalid.
    """
    config = model_config_from_spec(model_spec)
    provider_id = config.provider

    if provider_id not in _PROVIDER_REGISTRY:
        supported = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
        raise ValueError(f"Unknown provider '{provider_id}'. Supported providers: {supported}")

    factory = _PROVIDER_REGISTRY[provider_id]
    return factory(config, **kwargs)


def get_supported_providers() -> set[str]:
    """Return the registered provider ids."""
    return set(_PROVIDER_REGISTRY)


def register_provider(provider_id: str, factory: ProviderFactory) -> None:
    """Register a custom provider factory.

    Args:
        provider_id: The provider identifier (e.g., 'custom').
        factory: A callable that takes (model_id, **kwargs) and returns a Provider.
    """
    _PROVIDER_REGISTRY[provider_id] = factory
