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
        """
        Initialize the rate-limit error with an optional retry-after hint.
        
        Parameters:
            message (str): Human-readable error message describing the rate limit condition.
            retry_after (float | None): Number of seconds the caller may wait before retrying, or `None` if unspecified.
        """
        super().__init__(message)
        self.retry_after = retry_after


def _resolve_chat_completions_url(base_url: str) -> str:
    """
    Return a normalized URL for the OpenAI-compatible `/chat/completions` endpoint.
    
    Parameters:
        base_url (str): The base URL or endpoint. Trailing slashes are removed; if `base_url` already ends with `/chat/completions` it is returned unchanged.
    
    Returns:
        str: The normalized URL ending with `/chat/completions`.
    """
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
        """
        Initialize the OpenAI-compatible provider with model configuration and retry/backoff settings.
        
        Constructs internal configuration from the provided ModelConfig and optional overrides: resolves the base URL, determines the API key (from config, specified environment variable, or provided default), and stores retry/backoff parameters, jitter, and request timeout. Raises an error if an API key is required but cannot be found.
        
        Parameters:
            config (ModelConfig): Model and provider configuration used to populate defaults (base_url, api_key, api_key_env, provider name).
            default_base_url (str | None): Fallback base URL if not present in config.
            default_api_key_env (str | None): Fallback environment variable name to look up an API key if not present in config.
            require_api_key (bool): If True, raise ValueError when no API key is available.
            provider_name (str | None): Override name for the provider identity.
            max_retries (int): Maximum number of retry attempts for requests.
            base_delay (float): Base delay (seconds) used for exponential backoff.
            max_delay (float): Maximum delay (seconds) allowed for backoff.
            jitter (float): Fractional jitter applied to backoff delays.
            timeout (float): Request timeout in seconds.
        
        Raises:
            ValueError: When `require_api_key` is True and no API key can be found from config or environment.
        """
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
        """
        Configured model name for the provider.
        
        Returns:
            str: The name of the model configured for this provider.
        """
        return self._config.model

    @property
    def provider_id(self) -> str:
        """
        Provider identifier for this instance.
        
        Returns:
            str: The configured provider name.
        """
        return self._provider_name

    def _normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Normalize a list of message objects so each message's `content` is a list of simplified content entries.
        
        Parameters:
            messages (list[dict[str, Any]]): Message objects where each message may have a `content` field that is either a non-list value (left unchanged) or a list of content items. Content items expected are dicts with `type` equal to `"text"` (with `text` string) or `"image"` (with `image` string and optional `detail`).
        
        Returns:
            list[dict[str, Any]]: A new list of messages where messages whose `content` was a list are replaced with the same message but `content` set to a list of normalized entries:
                - Text items become {"type": "text", "text": <str>}.
                - Image items become {"type": "image_url", "image_url": {"url": <resolved URL>, "detail": <detail>}}.
            Items that are not dicts or that lack the expected fields are omitted; messages with non-list `content` are returned unchanged.
        """
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
        """
        Resolve an image reference into a URL suitable for API payloads.
        
        If `image_value` is already an HTTP(S) URL or a data URL, it is returned unchanged. If it is a filesystem path, the file is read and converted into a data URL (with a detected MIME type and base64-encoded content).
        
        Parameters:
            image_value (str): An image reference, either an HTTP(S) URL, a data URL, or a local filesystem path.
        
        Returns:
            str: An HTTP(S) URL or a data URL representing the image.
        """
        if image_value.startswith(("http://", "https://", "data:")):
            return image_value

        path = Path(image_value)
        mime_type = guess_type(path.as_posix())[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _get_headers(self) -> dict[str, str]:
        """
        Build HTTP headers for API requests, including configured headers and optional authorization.
        
        Merges the provider's configured headers with a default "Content-Type: application/json" and, when an API key is set, adds an "Authorization" header with a Bearer token.
        
        Returns:
            dict[str, str]: A mapping of header names to values for use in HTTP requests.
        """
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
        """
        Constructs the request payload for the OpenAI-compatible chat/completions endpoint.
        
        Parameters:
            prompt (str): Fallback user prompt used when `messages` is not provided.
            messages (list[dict[str, Any]] | None): Optional list of message objects; when provided, messages are normalized before inclusion. If omitted, a single user message with `prompt` is used.
            max_tokens (int | None): Optional maximum number of tokens to generate; included if provided.
            temperature (float | None): Optional sampling temperature; included if provided.
            **kwargs (Any): Additional payload fields that are merged into the final payload.
        
        Returns:
            dict[str, Any]: Payload containing at least `model` and `messages`, with `max_tokens`, `temperature`, options from the provider config, and any extra `kwargs` merged in.
        """
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
        """
        Compute the delay in seconds before the next retry, using a server-provided `Retry-After` value when available or exponential backoff with jitter.
        
        Parameters:
            attempt (int): The zero-based retry attempt count (0 for the first retry).
            retry_after (float | None): Optional server-specified retry delay in seconds.
        
        Returns:
            float: Seconds to wait before the next retry. If `retry_after` is provided, returns the lesser of that value and the provider's configured maximum delay; otherwise returns an exponential backoff delay (base_delay * 2**attempt) plus a random jitter, capped by the configured maximum delay.
        """
        if retry_after is not None:
            return float(min(retry_after, self._max_delay))
        delay = float(min(self._base_delay * (2**attempt), self._max_delay))
        jitter_amount = delay * self._jitter * random.random()
        return delay + jitter_amount

    def _parse_response(self, data: dict[str, Any]) -> Response:
        """
        Extracts the assistant's reply, normalized usage, and the original raw response from an OpenAI-compatible chat/completions response.
        
        Parameters:
            data (dict[str, Any]): Parsed JSON response returned by the chat/completions API.
        
        Returns:
            Response: An object containing:
                - text: the first choice's message content,
                - usage: the normalized usage information,
                - raw: the original response dict.
        
        Raises:
            ValueError: If the response contains no choices.
        """
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("No choices in response")
        text = choices[0].get("message", {}).get("content", "")
        usage = normalize_usage_dict(data.get("usage"))
        return Response(text=text, usage=usage, raw=data)

    def _handle_rate_limit(self, response: httpx.Response, attempt: int) -> float:
        """
        Compute a wait time after receiving a rate-limited response and log a warning.
        
        Parameters:
            response (httpx.Response): The HTTP response that triggered rate limiting; may contain a "Retry-After" header.
            attempt (int): Zero-based index of the current retry attempt.
        
        Returns:
            float: Number of seconds to wait before the next retry.
        """
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
        """
        Send a chat/completion request to the configured OpenAI-compatible API and return the parsed response.
        
        This method builds a request payload (using the provided prompt or extra kwargs), posts it to the provider's /chat/completions endpoint, and parses the first choice into a Response. It automatically retries on transient request errors and handles 429 rate-limit responses with backoff; if retries are exhausted, a OpenAICompatibleRateLimitError is raised.
        
        Parameters:
        	prompt (str): The prompt text to send when a message-based payload is not provided via kwargs.
        	max_tokens (int | None): Optional maximum number of tokens to generate.
        	temperature (float | None): Optional sampling temperature for generation.
        	**kwargs: Additional payload fields or provider-specific options (for example `messages`) merged into the request body.
        
        Returns:
        	Response: Parsed response containing the generated text, usage metrics, and raw response data.
        
        Raises:
        	OpenAICompatibleRateLimitError: If the request repeatedly fails due to rate limiting or transient errors and the retry limit is reached.
        """
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
        """
        Asynchronously send a chat/completions request to the configured OpenAI-compatible endpoint and return the parsed response.
        
        Parameters:
            prompt (str): The prompt string to use when messages are not provided.
            max_tokens (int | None): Optional maximum number of tokens to generate.
            temperature (float | None): Optional sampling temperature for generation.
            **kwargs (Any): Additional payload options merged into the request body (e.g., messages or provider-specific parameters).
        
        Returns:
            Response: Parsed response containing the generated text, usage information, and raw response data.
        
        Raises:
            OpenAICompatibleRateLimitError: If the request exhausts the configured retry attempts due to repeated failures or rate limiting.
        """
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
    """
    Create an OpenAI-compatible provider for chat/completions.
    
    Parameters:
        config (ModelConfig): Model configuration and defaults used by the provider.
        **kwargs: Additional provider options (e.g., api_key, base_url, max_retries, timeout) that override or extend `config`.
    
    Returns:
        OpenAICompatibleProvider: A provider instance configured to communicate with OpenAI-compatible chat/completions endpoints.
    """
    return OpenAICompatibleProvider(config, **kwargs)
