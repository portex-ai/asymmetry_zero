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
    """
    Create and return a Provider instance for the given model specification.
    
    Parameters:
        model_spec: A model specification — either a `ModelSpec`/ModelConfig object or a string like "provider:model".
        **kwargs: Additional keyword arguments forwarded to the provider factory.
    
    Returns:
        An initialized Provider corresponding to the provider identified by the model specification.
    
    Raises:
        ValueError: If the provider identifier from the model specification is not registered.
    """
    config = model_config_from_spec(model_spec)
    provider_id = config.provider

    if provider_id not in _PROVIDER_REGISTRY:
        supported = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
        raise ValueError(f"Unknown provider '{provider_id}'. Supported providers: {supported}")

    factory = _PROVIDER_REGISTRY[provider_id]
    return factory(config, **kwargs)


def get_supported_providers() -> set[str]:
    """
    List the currently registered provider identifiers.
    
    Returns:
        set[str]: A set containing the provider IDs registered in the provider registry.
    """
    return set(_PROVIDER_REGISTRY)


def register_provider(provider_id: str, factory: ProviderFactory) -> None:
    """
    Register a provider factory under the given identifier.
    
    Parameters:
        provider_id (str): Identifier used to look up the provider (e.g., "custom").
        factory (ProviderFactory): Callable that receives the resolved ModelConfig (and any extra kwargs passed to get_provider) and returns a Provider.
    """
    _PROVIDER_REGISTRY[provider_id] = factory
