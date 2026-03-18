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
        """
        Create and configure an OpenRouter provider instance, resolving API credentials and storing retry/backoff and timeout settings.
        
        The constructor resolves the API key from config.api_key if present, otherwise from the environment variable named by config.api_key_env (default "OPENROUTER_API_KEY"). It sets the provider API URL from config.base_url or the module default and stores retry/backoff parameters and request timeout. Raises ValueError if no API key can be found.
        
        Parameters:
            config (ModelConfig): Provider configuration containing model, optional api_key, api_key_env, base_url, headers, and options.
            max_retries (int): Maximum retry attempts for rate-limited requests.
            base_delay (float): Base delay in seconds used for exponential backoff.
            max_delay (float): Maximum delay in seconds to cap backoff.
            jitter (float): Fractional jitter (0–1) applied to backoff delays to randomize retries.
            timeout (float): Per-request timeout in seconds.
        
        Raises:
            ValueError: If an API key is not provided via config.api_key or the configured environment variable.
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
        """
        The configured model identifier for this provider.
        
        Returns:
            model_name (str): The model string from the provider configuration.
        """
        return self._config.model

    @property
    def provider_id(self) -> str:
        """
        Return the canonical identifier for this provider.
        
        Returns:
            provider_id (str): The provider identifier "openrouter".
        """
        return "openrouter"

    def _normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Normalize messages so each message's `content` is converted into a list of standardized content items.
        
        Processes each message in `messages`: if `message["content"]` is a list, items that are dicts with `type == "text"` produce `{"type": "text", "text": <str>}` entries, and items with `type == "image"` produce `{"type": "image_url", "image_url": {"url": <data-or-http-url>, "detail": <detail>}}` where the `url` value is obtained via self._image_url(...) and `detail` defaults to `"auto"`. Non-dict content items and unsupported types are ignored. Messages whose `content` is not a list are preserved unchanged.
        
        Parameters:
            messages (list[dict[str, Any]]): List of message objects to normalize. Each message is expected to be a mapping that may include a `content` key.
        
        Returns:
            list[dict[str, Any]]: A new list of messages with `content` fields normalized as described above.
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
        Convert an image reference into a URL suitable for embedding.
        
        Accepts an HTTP(s) URL, an existing data URL, or a local filesystem path. If given a local path, the file is read, its MIME type is inferred, and the content is encoded as a base64 data URL.
        
        Parameters:
            image_value (str): Image reference as an HTTP(s) URL, a data URL, or a local file path.
        
        Returns:
            str: The original `image_value` if it already starts with `http://`, `https://`, or `data:`, otherwise a `data:` URL containing the file's inferred MIME type and base64-encoded content.
        """
        if image_value.startswith(("http://", "https://", "data:")):
            return image_value

        path = Path(image_value)
        mime_type = guess_type(path.as_posix())[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _get_headers(self) -> dict[str, str]:
        """
        Construct the default HTTP headers used for OpenRouter API requests.
        
        Returns:
            headers (dict[str, str]): A mapping including Authorization (`Bearer <api_key>`), Content-Type (`application/json`), HTTP-Referer (`https://portex.ai`), X-Title (`portex-eval`), merged with any additional headers from the provider configuration.
        """
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
        """
        Build the request payload for an OpenRouter API call.
        
        Parameters:
            prompt (str): Fallback user prompt used when `messages` is not provided.
            messages (list[dict[str, Any]] | None): If a list is provided, it will be normalized and used as the `messages` field; otherwise `prompt` is wrapped into a single user message.
            max_tokens (int | None): Optional limit for the number of tokens to generate; included as `max_tokens` when provided.
            temperature (float | None): Optional sampling temperature; included as `temperature` when provided.
            **kwargs (Any): Additional payload fields that will be merged into the resulting payload, overriding values from `self._config.options` when keys conflict.
        
        Returns:
            dict[str, Any]: A payload dict containing at least `model` and `messages`, with optional `max_tokens`, `temperature`, merged `self._config.options`, and any extra fields from `kwargs`.
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
        """Calculate delay with exponential backoff and jitter."""
        if retry_after is not None:
            return float(min(retry_after, self._max_delay))
        delay = float(min(self._base_delay * (2**attempt), self._max_delay))
        jitter_amount = delay * self._jitter * random.random()
        return delay + jitter_amount

    def _parse_response(self, data: dict[str, Any]) -> Response:
        """
        Extract the primary text and usage information from an OpenRouter API response.
        
        Parameters:
            data (dict[str, Any]): Parsed JSON response from the OpenRouter API.
        
        Returns:
            Response: A Response object containing:
                - text: The content string from the first choice's message.
                - usage: A normalized usage dictionary derived from the response.
                - raw: The original response dictionary.
        
        Raises:
            ValueError: If the response contains no choices.
        """
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("No choices in response")
        text = choices[0].get("message", {}).get("content", "")
        usage = normalize_usage_dict(data.get("usage"))
        return Response(text=text, usage=usage, raw=data)

    def _handle_rate_limit(
        self,
        response: httpx.Response,
        attempt: int,
    ) -> float:
        """
        Compute the backoff delay to use after receiving a rate-limited response.
        
        If the response includes a `Retry-After` header and it can be parsed as a number, that value is considered when computing the delay; otherwise an exponential backoff with jitter based on `attempt` is used.
        
        Parameters:
            response (httpx.Response): The HTTP response that triggered the rate limit handling.
            attempt (int): Zero-based retry attempt count used to compute exponential backoff.
        
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
        """
        Generate a text completion using the configured OpenRouter model, retrying on rate limits and transient request errors.
        
        Parameters:
            prompt (str): The prompt or instruction to send to the model.
            max_tokens (int | None): Optional maximum number of tokens to generate.
            temperature (float | None): Sampling temperature; higher values produce more random output.
            **kwargs: Additional options merged into the request payload.
        
        Returns:
            Response: Parsed response containing the generated text, usage statistics, and raw API data.
        
        Raises:
            RateLimitError: If the provider exhausts the configured retry attempts without a successful response.
        """
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
        """
        Generate a model completion for the given prompt using the provider's configuration and retry on rate limits.
        
        The method builds a request payload from the prompt and optional generation parameters, sends it to the configured OpenRouter endpoint, and will retry with backoff when encountering rate-limit responses. Raises RateLimitError if the allowed retry attempts are exhausted.
        
        Parameters:
            prompt (str): The text prompt to generate a completion for.
            max_tokens (int | None): Maximum number of tokens to generate. If None, provider defaults apply.
            temperature (float | None): Sampling temperature to control randomness. If None, provider defaults apply.
            **kwargs: Additional options forwarded into the request payload (merged with provider config options).
        
        Returns:
            Response: Parsed response object containing the generated text, normalized usage metrics, and raw API response data.
        
        Raises:
            RateLimitError: If the provider exhausts the configured retry attempts due to rate limiting.
        """
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
    """
    Create an OpenRouterProvider configured with the given ModelConfig.
    
    Parameters:
        config: Resolved ModelConfig containing the model identifier and any provider-specific options (API key source, base URL, headers, etc.).
        **kwargs: Additional keyword arguments forwarded to OpenRouterProvider constructor.
    
    Returns:
        An OpenRouterProvider configured using the provided config and kwargs.
    """
    return OpenRouterProvider(config, **kwargs)
