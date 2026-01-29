"""Custom solver wrappers for role tagging and provider integration."""

from __future__ import annotations

import os
from typing import Any, Literal

from inspect_ai.model import ChatMessage, ChatMessageAssistant, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver

from portex_eval.providers import Provider, get_provider


def _get_candidate_provider(model_string: str | None = None) -> Provider | None:
    """Get a provider instance for the candidate model.

    Args:
        model_string: Optional model string (e.g., 'openrouter:google/gemini-2.5-flash').
            If not provided, tries PORTEX_CANDIDATE_MODEL env var.

    Returns:
        Provider instance or None if no model string provided.
    """
    model_string = model_string or os.environ.get("PORTEX_CANDIDATE_MODEL")
    if not model_string:
        return None
    return get_provider(model_string)


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
            response = await provider.agenerate(prompt, **kwargs)
            state.messages.append(ChatMessageAssistant(content=response.text))
            state.output = type("Output", (), {"completion": response.text})()
            return state

        model = get_model()
        if model is not None and model.role != "candidate":
            model._set_role("candidate")
        return await generate(state, tool_calls=tool_calls, **kwargs)

    return solve


@solver
def provider_generate(
    model_string: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
    **kwargs: Any,
) -> Solver:
    """Generate using a portex_eval provider.

    This solver bypasses Inspect's model system and directly uses the
    portex_eval provider abstraction, which includes rate limit handling
    with exponential backoff.

    Args:
        model_string: Model string (e.g., 'openrouter:google/gemini-2.5-flash').
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        **kwargs: Additional arguments passed to provider.generate().

    Returns:
        Solver function.
    """
    provider = get_provider(model_string)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        prompt = _extract_prompt_from_messages(state.messages)
        response = await provider.agenerate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        state.messages.append(ChatMessageAssistant(content=response.text))
        state.output = type("Output", (), {"completion": response.text})()
        return state

    return solve
