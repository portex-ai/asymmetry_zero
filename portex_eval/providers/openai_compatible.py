"""Shared OpenAI-compatible provider implementation."""

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

DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0
DEFAULT_JITTER = 0.5
DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAICompatibleRateLimitError(Exception):
    """Raised when retries are exhausted after rate limiting."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def _resolve_chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


class OpenAICompatibleProvider(Provider):
    """Provider for OpenAI-compatible ``/chat/completions`` APIs."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        default_base_url: str | None = None,
        default_api_key_env: str | None = None,
        require_api_key: bool = False,
        provider_name: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        jitter: float = DEFAULT_JITTER,
        timeout: float = 120.0,
    ):
        self._config = config
        self._provider_name = provider_name or config.provider
        self._base_url = config.base_url or default_base_url or DEFAULT_BASE_URL
        env_var = config.api_key_env or default_api_key_env
        self._api_key = config.api_key or (os.environ.get(env_var) if env_var else None)
        if require_api_key and not self._api_key:
            env_msg = f" Set {env_var} or pass api_key." if env_var else ""
            raise ValueError(f"{self._provider_name} API key not found.{env_msg}")
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
        return self._provider_name

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
        headers = {"Content-Type": "application/json", **self._config.headers}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

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
        if retry_after is not None:
            return float(min(retry_after, self._max_delay))
        delay = float(min(self._base_delay * (2**attempt), self._max_delay))
        jitter_amount = delay * self._jitter * random.random()
        return delay + jitter_amount

    def _parse_response(self, data: dict[str, Any]) -> Response:
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("No choices in response")
        content = choices[0].get("message", {}).get("content")
        text = content if isinstance(content, str) else ""
        usage = normalize_usage_dict(data.get("usage"))
        return Response(text=text, usage=usage, raw=data)

    def _handle_rate_limit(self, response: httpx.Response, attempt: int) -> float:
        retry_after = None
        if "Retry-After" in response.headers:
            try:
                retry_after = float(response.headers["Retry-After"])
            except ValueError:
                pass
        delay = self._calculate_delay(attempt, retry_after)
        logger.warning(
            "Rate limited by %s (attempt %s/%s). Waiting %.1fs before retry.",
            self._provider_name,
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
        url = _resolve_chat_completions_url(self._base_url)

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
                    "Request error for %s (attempt %s/%s): %s. Waiting %.1fs before retry.",
                    self._provider_name,
                    attempt + 1,
                    self._max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
                continue

        raise OpenAICompatibleRateLimitError(
            f"Max retries ({self._max_retries}) exceeded for {self._provider_name}",
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
        url = _resolve_chat_completions_url(self._base_url)

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
                    "Request error for %s (attempt %s/%s): %s. Waiting %.1fs before retry.",
                    self._provider_name,
                    attempt + 1,
                    self._max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                continue

        raise OpenAICompatibleRateLimitError(
            f"Max retries ({self._max_retries}) exceeded for {self._provider_name}",
        ) from last_error


def create_openai_compatible_provider(
    config: ModelConfig,
    **kwargs: Any,
) -> OpenAICompatibleProvider:
    """Create a generic OpenAI-compatible provider."""
    return OpenAICompatibleProvider(config, **kwargs)
