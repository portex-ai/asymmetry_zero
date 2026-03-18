"""Anthropic provider implementation."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any

import httpx

from portex_eval.providers.base import ModelConfig, Provider, Response

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0
DEFAULT_JITTER = 0.5
DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 1024


class AnthropicRateLimitError(Exception):
    """Raised when retries are exhausted after rate limiting."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def _resolve_messages_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/messages"):
        return base
    return f"{base}/messages"


class AnthropicProvider(Provider):
    """Provider for Anthropic ``/messages`` APIs."""

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
        self._config = config
        self._base_url = config.base_url or os.environ.get("ANTHROPIC_BASE_URL", DEFAULT_BASE_URL)
        env_var = config.api_key_env or "ANTHROPIC_API_KEY"
        self._api_key = config.api_key or os.environ.get(env_var)
        if not self._api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY or pass api_key.")
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
        return "anthropic"

    def _get_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": DEFAULT_ANTHROPIC_VERSION,
            "content-type": "application/json",
            **self._config.headers,
        }

    def _build_payload(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        payload.update(self._config.options)
        payload.update(kwargs)
        return payload

    def _calculate_delay(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return float(min(retry_after, self._max_delay))
        delay = float(min(self._base_delay * (2**attempt), self._max_delay))
        jitter_amount = delay * self._jitter * random.random()
        return delay + jitter_amount

    def _parse_response(self, data: dict[str, Any]) -> Response:
        content = data.get("content", [])
        text_parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return Response(text="".join(text_parts), usage=data.get("usage"), raw=data)

    def _handle_rate_limit(self, response: httpx.Response, attempt: int) -> float:
        retry_after = None
        if "Retry-After" in response.headers:
            try:
                retry_after = float(response.headers["Retry-After"])
            except ValueError:
                pass
        delay = self._calculate_delay(attempt, retry_after)
        logger.warning(
            "Rate limited by Anthropic (attempt %s/%s). Waiting %.1fs before retry.",
            attempt + 1,
            self._max_retries,
            delay,
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
        payload = self._build_payload(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        headers = self._get_headers()
        url = _resolve_messages_url(self._base_url)

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(url, json=payload, headers=headers)

                if response.status_code == 429:
                    delay = self._handle_rate_limit(response, attempt)
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                return self._parse_response(response.json())
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    delay = self._handle_rate_limit(exc.response, attempt)
                    time.sleep(delay)
                    continue
                raise
            except httpx.RequestError as exc:
                last_error = exc
                delay = self._calculate_delay(attempt)
                logger.warning(
                    "Request error for Anthropic (attempt %s/%s): %s. Waiting %.1fs before retry.",
                    attempt + 1,
                    self._max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
                continue

        raise AnthropicRateLimitError(
            f"Max retries ({self._max_retries}) exceeded for Anthropic",
        ) from last_error

    async def agenerate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Response:
        payload = self._build_payload(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        headers = self._get_headers()
        url = _resolve_messages_url(self._base_url)

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)

                if response.status_code == 429:
                    delay = self._handle_rate_limit(response, attempt)
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()
                return self._parse_response(response.json())
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    delay = self._handle_rate_limit(exc.response, attempt)
                    await asyncio.sleep(delay)
                    continue
                raise
            except httpx.RequestError as exc:
                last_error = exc
                delay = self._calculate_delay(attempt)
                logger.warning(
                    "Request error for Anthropic (attempt %s/%s): %s. Waiting %.1fs before retry.",
                    attempt + 1,
                    self._max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                continue

        raise AnthropicRateLimitError(
            f"Max retries ({self._max_retries}) exceeded for Anthropic",
        ) from last_error


def create_anthropic_provider(config: ModelConfig, **kwargs: Any) -> AnthropicProvider:
    """Factory for ``anthropic:<model>`` specs."""
    return AnthropicProvider(config, **kwargs)
