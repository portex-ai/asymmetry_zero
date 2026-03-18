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
        super().__init__(
            config,
            default_base_url=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
            default_api_key_env="OPENAI_API_KEY",
            require_api_key=True,
            provider_name="openai",
            **kwargs,
        )


def create_openai_provider(config: ModelConfig, **kwargs: Any) -> OpenAIProvider:
    """Factory for ``openai:<model>`` specs."""
    return OpenAIProvider(config, **kwargs)
