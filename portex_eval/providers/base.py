"""Abstract base classes and config helpers for model providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeAlias


@dataclass
class Response:
    """Standard response from a provider generate call."""

    text: str
    usage: dict[str, int] | None = None
    raw: Any = None


@dataclass(frozen=True)
class ModelConfig:
    """Resolved provider configuration for one model endpoint."""

    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def model_string(self) -> str:
        """Return the canonical ``provider:model`` string form."""
        return f"{self.provider}:{self.model}"


ModelSpec: TypeAlias = str | ModelConfig | dict[str, Any]


class Provider(ABC):
    """Abstract base class for model providers.

    Providers wrap external LLM APIs and expose a uniform interface
    for generation calls used by the benchmark runner.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Response:
        """Generate a completion from the model.

        Args:
            prompt: The input prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Provider-specific options.

        Returns:
            A Response with the generated text and optional metadata.
        """
        ...

    @abstractmethod
    async def agenerate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Response:
        """Async version of generate."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model identifier (e.g., 'google/gemini-2.5-flash')."""
        ...

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """The provider identifier (e.g., 'openrouter')."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r})"


def parse_model_string(model_string: str) -> tuple[str, str]:
    """Parse a model string of format 'provider:model_id'.

    Args:
        model_string: A string like 'openrouter:google/gemini-2.5-flash'.

    Returns:
        A tuple of (provider_id, model_id).

    Raises:
        ValueError: If the format is invalid.
    """
    if ":" not in model_string:
        raise ValueError(
            f"Invalid model string '{model_string}'. "
            "Expected format: 'provider:model_id' (e.g., 'openrouter:google/gemini-2.5-flash')"
        )
    provider_id, model_id = model_string.split(":", 1)
    if not provider_id or not model_id:
        raise ValueError(
            f"Invalid model string '{model_string}'. Both provider and model_id must be non-empty."
        )
    return provider_id, model_id


def model_config_from_spec(spec: ModelSpec) -> ModelConfig:
    """Normalize a model spec into a ``ModelConfig``."""
    if isinstance(spec, ModelConfig):
        return spec

    if isinstance(spec, str):
        provider_id, model_id = parse_model_string(spec)
        return ModelConfig(provider=provider_id, model=model_id)

    if not isinstance(spec, dict):
        raise ValueError(
            "Model spec must be a string like 'provider:model' or a config object."
        )

    provider_id = spec.get("provider")
    model_id = spec.get("model") or spec.get("model_id")
    if not isinstance(provider_id, str) or not provider_id.strip():
        raise ValueError("Model config must include non-empty string field 'provider'.")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Model config must include non-empty string field 'model'.")

    base_url = spec.get("base_url")
    api_key = spec.get("api_key")
    api_key_env = spec.get("api_key_env")
    headers = spec.get("headers") or {}
    options = spec.get("options") or {}

    if base_url is not None and not isinstance(base_url, str):
        raise ValueError("Model config field 'base_url' must be a string when provided.")
    if api_key is not None and not isinstance(api_key, str):
        raise ValueError("Model config field 'api_key' must be a string when provided.")
    if api_key_env is not None and not isinstance(api_key_env, str):
        raise ValueError("Model config field 'api_key_env' must be a string when provided.")
    if not isinstance(headers, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in headers.items()
    ):
        raise ValueError("Model config field 'headers' must be an object of string pairs.")
    if not isinstance(options, dict):
        raise ValueError("Model config field 'options' must be an object when provided.")

    return ModelConfig(
        provider=provider_id.strip(),
        model=model_id.strip(),
        base_url=base_url.strip() if isinstance(base_url, str) else None,
        api_key=api_key,
        api_key_env=api_key_env,
        headers=dict(headers),
        options=dict(options),
    )


def model_config_to_dict(config: ModelConfig) -> dict[str, Any]:
    """Serialize a ``ModelConfig`` into a JSON-safe dictionary."""
    payload: dict[str, Any] = {
        "provider": config.provider,
        "model": config.model,
    }
    if config.base_url:
        payload["base_url"] = config.base_url
    if config.api_key:
        payload["api_key"] = config.api_key
    if config.api_key_env:
        payload["api_key_env"] = config.api_key_env
    if config.headers:
        payload["headers"] = dict(config.headers)
    if config.options:
        payload["options"] = dict(config.options)
    return payload


def normalize_usage_dict(usage: Any) -> dict[str, int] | None:
    """Normalize provider usage payloads to input/output/total token keys."""
    if not isinstance(usage, dict):
        return None

    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        total_tokens = int(input_tokens) + int(output_tokens)

    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
    }
