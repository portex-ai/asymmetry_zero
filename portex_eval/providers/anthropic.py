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
        """
        Initialize the AnthropicRateLimitError with an error message and optional retry-after duration.
        
        Parameters:
            message (str): Human-readable error message describing the rate limit condition.
            retry_after (float | None): Seconds the caller should wait before retrying, or `None` if unknown.
        """
        super().__init__(message)
        self.retry_after = retry_after


def _resolve_messages_url(base_url: str) -> str:
    """
    Normalize a base URL and ensure it points to the Anthropic `/messages` endpoint.
    
    Parameters:
        base_url (str): Base Anthropic API URL or host; may include or omit trailing slash or the `/messages` path.
    
    Returns:
        str: The normalized URL that ends with `/messages` with no duplicate trailing slashes.
    """
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
        """
        Initialize the provider with model configuration and retry/backoff settings.
        
        Parameters:
            config (ModelConfig): Model configuration including model name and optional API key or API key environment variable and base URL.
            max_retries (int): Maximum number of retry attempts for transient failures and rate limits.
            base_delay (float): Initial delay in seconds used for exponential backoff.
            max_delay (float): Maximum delay in seconds to cap backoff waits.
            jitter (float): Maximum random jitter in seconds added to backoff delays.
            timeout (float): Request timeout in seconds for HTTP calls.
        
        Raises:
            ValueError: If no Anthropic API key is found in the provided config or the configured environment variable.
        """
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
        """
        Get the configured model name for this provider.
        
        Returns:
            The model identifier from the provider configuration.
        """
        return self._config.model

    @property
    def provider_id(self) -> str:
        """
        Return the provider identifier for this implementation.
        
        Returns:
            str: The static provider identifier "anthropic".
        """
        return "anthropic"

    def _get_headers(self) -> dict[str, str]:
        """
        Builds the HTTP headers used for requests to the Anthropic /messages API.
        
        Returns:
            dict[str, str]: Mapping of header names to values including `x-api-key` (the configured API key), `anthropic-version`, and `content-type`, merged with any extra headers from the provider config (config headers may override the defaults).
        """
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
        """
        Construct the JSON payload for an Anthropic /messages request.
        
        Builds a payload containing the configured model, a single user message with `prompt`, and `max_tokens` (uses DEFAULT_MAX_TOKENS when `max_tokens` is None). Includes `temperature` when provided, then merges provider configuration options and any additional keyword arguments; later keys override earlier ones.
        
        Parameters:
            prompt: The user-facing prompt text to send as the message content.
            max_tokens: Maximum tokens to generate; if omitted, the provider's default `DEFAULT_MAX_TOKENS` is used.
            temperature: Sampling temperature to include in the payload when provided.
            **kwargs: Additional payload fields to merge into the final payload (overrides config options when keys collide).
        
        Returns:
            A dict representing the request payload ready to be sent to Anthropic's /messages endpoint.
        """
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
        """
        Compute the retry delay in seconds, using Retry-After when provided or exponential backoff with jitter otherwise.
        
        If `retry_after` is set, the returned delay is the smaller of `retry_after` and the provider's configured maximum delay. Otherwise the delay is `base_delay * 2**attempt` capped at the maximum delay, plus a random jitter up to `jitter` fraction of that delay.
        
        Parameters:
            attempt (int): Zero-based retry attempt count (0 for first retry).
            retry_after (float | None): Optional server-suggested delay in seconds.
        
        Returns:
            float: Delay in seconds to wait before the next retry.
        """
        if retry_after is not None:
            return float(min(retry_after, self._max_delay))
        delay = float(min(self._base_delay * (2**attempt), self._max_delay))
        jitter_amount = delay * self._jitter * random.random()
        return delay + jitter_amount

    def _parse_response(self, data: dict[str, Any]) -> Response:
        """
        Extracts and returns the textual output and metadata from an Anthropic /messages response payload.
        
        Parameters:
            data (dict[str, Any]): Response JSON expected to contain a "content" list of items (each item may be a dict with at least "type" and "text" keys) and an optional "usage" entry.
        
        Returns:
            Response: A Response object whose `text` is the concatenation of all content item `text` values with type "text", `usage` set from `data["usage"]` if present, and `raw` containing the original `data`.
        """
        content = data.get("content", [])
        text_parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return Response(text="".join(text_parts), usage=data.get("usage"), raw=data)

    def _handle_rate_limit(self, response: httpx.Response, attempt: int) -> float:
        """
        Compute how long to wait before retrying after a rate-limited response.
        
        Reads the response's `Retry-After` header (if present and parsable) and delegates to
        _internal backoff calculation to produce a delay. Logs a warning with attempt and
        delay information.
        
        Parameters:
            response (httpx.Response): HTTP response received from Anthropic; used to read `Retry-After`.
            attempt (int): Zero-based retry attempt index used to compute the backoff.
        
        Returns:
            float: Delay in seconds to wait before the next retry.
        """
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
        """
        Generate text for the given prompt using the configured Anthropic /messages endpoint, automatically handling retries, rate limits (429), and transient request errors.
        
        Parameters:
            prompt (str): The user prompt to send to the model.
            max_tokens (int | None): Maximum number of tokens to generate; uses the provider's default when None.
            temperature (float | None): Sampling temperature; uses the provider's default when None.
            **kwargs: Additional payload options merged into the request body.
        
        Returns:
            Response: Parsed response containing generated text, usage information, and raw API data.
        
        Raises:
            AnthropicRateLimitError: If the provider exhausts its retry attempts (may chain the last request error).
        """
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
        """
        Asynchronously generate a completion from the Anthropic /messages API for the given prompt.
        
        Parameters:
            prompt (str): The prompt text to send to the model.
            max_tokens (int | None): Maximum tokens for the response; uses the provider's default when None.
            temperature (float | None): Sampling temperature to control randomness; omitted when None.
            **kwargs: Additional options merged into the request payload.
        
        Returns:
            Response: Parsed response containing the concatenated text, usage metrics, and raw API data.
        
        Raises:
            AnthropicRateLimitError: If the provider exhausts all retry attempts (e.g., due to persistent rate limiting or repeated request failures).
        """
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
    """
    Create an AnthropicProvider configured for the given model configuration.
    
    Parameters:
        config (ModelConfig): Model configuration used to initialize the provider. Additional keyword arguments are forwarded to AnthropicProvider.
    
    Returns:
        AnthropicProvider: A provider instance configured for the specified model.
    """
    return AnthropicProvider(config, **kwargs)
