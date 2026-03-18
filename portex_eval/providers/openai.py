"""OpenAI provider implementation."""

from __future__ import annotations

import os
from typing import Any

from portex_eval.providers.base import ModelConfig
from portex_eval.providers.openai_compatible import (
    DEFAULT_BASE_URL,
    OpenAICompatibleProvider,
)


class OpenAIProvider(OpenAICompatibleProvider):
    """Provider for OpenAI chat completions."""

    def __init__(self, config: ModelConfig, **kwargs: Any):
        """
        Initialize the OpenAI provider using the given model configuration and sensible OpenAI-specific defaults.
        
        Parameters:
            config (ModelConfig): Model configuration for the provider.
            **kwargs: Additional keyword arguments forwarded to the base OpenAICompatibleProvider.
        
        Notes:
            - The base URL defaults to the value of the OPENAI_BASE_URL environment variable if set, otherwise DEFAULT_BASE_URL.
            - The provider uses "OPENAI_API_KEY" as the default API key environment variable and requires an API key.
            - The provider is identified as "openai".
        """
        super().__init__(
            config,
            default_base_url=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
            default_api_key_env="OPENAI_API_KEY",
            require_api_key=True,
            provider_name="openai",
            **kwargs,
        )


def create_openai_provider(config: ModelConfig, **kwargs: Any) -> OpenAIProvider:
    """
    Create an OpenAIProvider for an "openai:<model>" specification.
    
    Parameters:
        config (ModelConfig): Model configuration describing the target OpenAI model.
        **kwargs: Additional keyword arguments forwarded to OpenAIProvider.
    
    Returns:
        OpenAIProvider: A provider instance configured for the specified model.
    """
    return OpenAIProvider(config, **kwargs)
