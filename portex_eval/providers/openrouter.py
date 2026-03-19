"""OpenRouter provider implementation."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import random
import time
from mimetypes import guess_type
from pathlib import Path
from typing import Any

import httpx

from portex_eval.providers.base import ModelConfig, Provider, Response, normalize_usage_dict

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Rate limit retry configuration
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0
DEFAULT_JITTER = 0.5


class RateLimitError(Exception):
    """Raised when rate limit is hit and retries are exhausted."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class OpenRouterProvider(Provider):
    """Provider for OpenRouter API.

    Supports model string format: openrouter:<model_id>
    Example: openrouter:google/gemini-2.5-flash

    Reads OPENROUTER_API_KEY from environment.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        jitter: float = DEFAULT_JITTER,
        timeout: float = 120.0,
    ):
        """Initialize the OpenRouter provider.

        Args:
            config: Resolved provider configuration.
            max_retries: Maximum number of retries on rate limit.
            base_delay: Base delay in seconds for exponential backoff.
            max_delay: Maximum delay in seconds between retries.
            jitter: Jitter factor (0-1) to add randomness to delays.
            timeout: Request timeout in seconds.
        """
        self._config = config
        env_var = config.api_key_env or "OPENROUTER_API_KEY"
        self._api_key = config.api_key or os.environ.get(env_var)
        if not self._api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                f"Set {env_var} environment variable or pass api_key."
            )
        self._api_url = config.base_url or OPENROUTER_API_URL
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._jitter = jitter
        self._timeout = timeout

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def provider_id(self) -> str:
        return "openrouter"

    def _normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                normalized.append(message)
                continue

            parts: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append({"type": "text", "text": text})
                elif item_type == "image":
                    image_value = item.get("image")
                    detail = item.get("detail", "auto")
                    if isinstance(image_value, str):
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": self._image_url(image_value),
                                    "detail": detail,
                                },
                            }
                        )
            normalized.append({**message, "content": parts})
        return normalized

    def _image_url(self, image_value: str) -> str:
        if image_value.startswith(("http://", "https://", "data:")):
            return image_value

        path = Path(image_value)
        mime_type = guess_type(path.as_posix())[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://portex.ai",
            "X-Title": "portex-eval",
            **self._config.headers,
        }

    def _build_payload(
        self,
        prompt: str,
        *,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": (
                self._normalize_messages(messages)
                if isinstance(messages, list)
                else [{"role": "user", "content": prompt}]
            ),
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        payload.update(self._config.options)
        payload.update(kwargs)
        return payload

    def _calculate_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Calculate delay with exponential backoff and jitter."""
        if retry_after is not None:
            return float(min(retry_after, self._max_delay))
        delay = float(min(self._base_delay * (2**attempt), self._max_delay))
        jitter_amount = delay * self._jitter * random.random()
        return delay + jitter_amount

    def _parse_response(self, data: dict[str, Any]) -> Response:
        """Parse OpenRouter API response."""
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("No choices in response")
        content = choices[0].get("message", {}).get("content")
        text = content if isinstance(content, str) else ""
        usage = normalize_usage_dict(data.get("usage"))
        return Response(text=text, usage=usage, raw=data)

    def _handle_rate_limit(
        self,
        response: httpx.Response,
        attempt: int,
    ) -> float:
        """Handle rate limit response, return delay to wait."""
        retry_after = None
        if "Retry-After" in response.headers:
            try:
                retry_after = float(response.headers["Retry-After"])
            except ValueError:
                pass
        delay = self._calculate_delay(attempt, retry_after)
        logger.warning(
            f"Rate limited by OpenRouter (attempt {attempt + 1}/{self._max_retries}). "
            f"Waiting {delay:.1f}s before retry."
        )
        return delay

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Response:
        """Generate a completion synchronously with rate limit retry."""
        payload = self._build_payload(
            prompt, max_tokens=max_tokens, temperature=temperature, **kwargs
        )
        headers = self._get_headers()

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(
                        self._api_url,
                        json=payload,
                        headers=headers,
                    )

                if response.status_code == 429:
                    delay = self._handle_rate_limit(response, attempt)
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                return self._parse_response(response.json())

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    delay = self._handle_rate_limit(e.response, attempt)
                    time.sleep(delay)
                    continue
                raise

            except httpx.RequestError as e:
                last_error = e
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Request error (attempt {attempt + 1}/{self._max_retries}): {e}. "
                    f"Waiting {delay:.1f}s before retry."
                )
                time.sleep(delay)
                continue

        raise RateLimitError(
            f"Max retries ({self._max_retries}) exceeded for OpenRouter API",
        ) from last_error

    async def agenerate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Response:
        """Generate a completion asynchronously with rate limit retry."""
        payload = self._build_payload(
            prompt, max_tokens=max_tokens, temperature=temperature, **kwargs
        )
        headers = self._get_headers()

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        self._api_url,
                        json=payload,
                        headers=headers,
                    )

                if response.status_code == 429:
                    delay = self._handle_rate_limit(response, attempt)
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()
                return self._parse_response(response.json())

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    delay = self._handle_rate_limit(e.response, attempt)
                    await asyncio.sleep(delay)
                    continue
                raise

            except httpx.RequestError as e:
                last_error = e
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Request error (attempt {attempt + 1}/{self._max_retries}): {e}. "
                    f"Waiting {delay:.1f}s before retry."
                )
                await asyncio.sleep(delay)
                continue

        raise RateLimitError(
            f"Max retries ({self._max_retries}) exceeded for OpenRouter API",
        ) from last_error


def create_openrouter_provider(
    config: ModelConfig,
    **kwargs: Any,
) -> OpenRouterProvider:
    """Factory function to create an OpenRouter provider.

    Args:
        config: Resolved model config.
        **kwargs: Additional arguments passed to OpenRouterProvider.

    Returns:
        An initialized OpenRouterProvider instance.
    """
    return OpenRouterProvider(config, **kwargs)
