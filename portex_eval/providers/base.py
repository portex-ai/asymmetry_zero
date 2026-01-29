"""Abstract base class for model providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Response:
    """Standard response from a provider generate call."""

    text: str
    usage: dict[str, int] | None = None
    raw: Any = None


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
