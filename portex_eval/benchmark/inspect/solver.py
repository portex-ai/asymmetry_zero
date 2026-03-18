"""Custom solver wrappers for role tagging and provider integration."""

from __future__ import annotations

import json
import os
from mimetypes import guess_type
from typing import Any, Literal

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    Logprob,
    Logprobs,
    ModelOutput,
    ModelUsage,
    TopLogprob,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

from portex_eval.providers import ModelSpec, Provider, get_provider


def _candidate_provider_spec(model_spec: ModelSpec | None = None) -> ModelSpec | None:
    if model_spec is not None:
        return model_spec
    config_text = os.environ.get("PORTEX_CANDIDATE_CONFIG")
    if config_text:
        return json.loads(config_text)
    return os.environ.get("PORTEX_CANDIDATE_MODEL")


def _get_candidate_provider(model_spec: ModelSpec | None = None) -> Provider | None:
    """Get a provider instance for the candidate model.

    Args:
        model_spec: Optional model spec. If not provided, loads from env.

    Returns:
        Provider instance or None if no model string provided.
    """
    model_spec = _candidate_provider_spec(model_spec)
    if not model_spec:
        return None
    return get_provider(model_spec)


def _extract_prompt_from_messages(messages: list[ChatMessage]) -> str:
    """Extract text prompt from chat messages for provider API."""
    parts = []
    for msg in messages:
        if hasattr(msg, "content"):
            if isinstance(msg.content, str):
                parts.append(msg.content)
            elif isinstance(msg.content, list):
                for item in msg.content:
                    if hasattr(item, "text"):
                        parts.append(item.text)
    return "\n".join(parts)


def _provider_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for message in messages:
        role = getattr(message, "role", None)
        if role not in {"system", "user", "assistant"}:
            continue

        content = getattr(message, "content", None)
        if isinstance(content, str):
            serialized.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            continue

        parts: list[dict[str, Any]] = []
        for item in content:
            item_type = getattr(item, "type", None)
            if item_type == "text":
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append({"type": "text", "text": text})
            elif item_type == "image":
                image = getattr(item, "image", None)
                detail = getattr(item, "detail", None) or "auto"
                if isinstance(image, str):
                    parts.append({"type": "image", "image": image, "detail": detail})
            elif item_type == "document":
                document = getattr(item, "document", None)
                if isinstance(document, str):
                    mime_type = guess_type(document)[0] or "application/octet-stream"
                    parts.append(
                        {"type": "text", "text": f"[document: {document} ({mime_type})]"}
                    )

        serialized.append({"role": role, "content": parts})
    return serialized


def _provider_logprobs(raw: Any) -> Logprobs | None:
    if not isinstance(raw, dict):
        return None
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    if not isinstance(content, list):
        return None

    entries: list[Logprob] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        top_candidates = item.get("top_logprobs")
        entries.append(
            Logprob(
                token=str(item.get("token", "")),
                logprob=float(item.get("logprob", 0.0)),
                bytes=item.get("bytes"),
                top_logprobs=(
                    [
                        TopLogprob(
                            token=str(candidate.get("token", "")),
                            logprob=float(candidate.get("logprob", 0.0)),
                            bytes=candidate.get("bytes"),
                        )
                        for candidate in top_candidates
                        if isinstance(candidate, dict)
                    ]
                    if isinstance(top_candidates, list)
                    else None
                ),
            )
        )
    return Logprobs(content=entries)


def _provider_model_output(
    provider: Provider,
    text: str,
    usage: dict[str, int] | None,
    raw: Any,
) -> ModelOutput:
    output = ModelOutput.from_content(
        model=f"{provider.provider_id}:{provider.model_name}",
        content=text,
    )
    if usage:
        output.usage = ModelUsage(
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
        )
    if output.choices:
        output.choices[0].logprobs = _provider_logprobs(raw)
    return output


@solver
def candidate_generate(
    model_string: str | None = None,
    tool_calls: Literal["loop", "single", "none"] = "loop",
    use_provider: bool = False,
    **kwargs: Any,
) -> Solver:
    """Generate with the candidate model role for logging.

    This solver can operate in two modes:
    1. Standard Inspect mode (use_provider=False): Uses Inspect's built-in model handling
    2. Provider mode (use_provider=True): Uses the portex_eval provider abstraction

    Args:
        model_string: Optional model string (e.g., 'openrouter:google/gemini-2.5-flash').
            Required if use_provider=True and PORTEX_CANDIDATE_MODEL not set.
        tool_calls: Tool call handling mode ('loop', 'single', 'none').
        use_provider: If True, use portex_eval.providers for generation instead
            of Inspect's model system. Enables rate limit retry with backoff.
        **kwargs: Additional arguments passed to generate or provider.

    Returns:
        Solver function.
    """
    provider: Provider | None = None

    if use_provider:
        provider = _get_candidate_provider(model_string)
        if provider is None:
            raise ValueError(
                "model_string is required when use_provider=True. "
                "Provide it directly or set PORTEX_CANDIDATE_MODEL env var."
            )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if use_provider and provider is not None:
            prompt = _extract_prompt_from_messages(state.messages)
            response = await provider.agenerate(
                prompt,
                messages=_provider_messages(state.messages),
                **kwargs,
            )
            state.output = _provider_model_output(
                provider,
                response.text,
                response.usage,
                response.raw,
            )
            state.messages.append(state.output.message)
            return state

        model = get_model()
        if model is not None and model.role != "candidate":
            model._set_role("candidate")
        return await generate(state, tool_calls=tool_calls, **kwargs)

    return solve


@solver
def provider_generate(
    model_spec: ModelSpec,
    max_tokens: int | None = None,
    temperature: float | None = None,
    **kwargs: Any,
) -> Solver:
    """Generate using a portex_eval provider.

    This solver bypasses Inspect's model system and directly uses the
    portex_eval provider abstraction, which includes rate limit handling
    with exponential backoff.

    Args:
        model_spec: Model string or config object.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        **kwargs: Additional arguments passed to provider.generate().

    Returns:
        Solver function.
    """
    provider = get_provider(model_spec)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        prompt = _extract_prompt_from_messages(state.messages)
        response = await provider.agenerate(
            prompt,
            messages=_provider_messages(state.messages),
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        state.output = _provider_model_output(provider, response.text, response.usage, response.raw)
        state.messages.append(state.output.message)
        return state

    return solve
