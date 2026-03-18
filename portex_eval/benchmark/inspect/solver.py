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
    """
    Resolve the candidate ModelSpec using an explicit argument, a JSON config env var, or a fallback env var.
    
    If `model_spec` is provided, it is returned unchanged. Otherwise, the function attempts to parse the environment variable `PORTEX_CANDIDATE_CONFIG` as JSON and return its value. If that variable is absent or empty, the function returns the value of the `PORTEX_CANDIDATE_MODEL` environment variable (or `None` if neither env var is set).
    
    Parameters:
        model_spec (ModelSpec | None): An explicit model specification to use instead of reading environment variables.
    
    Returns:
        ModelSpec | None: The resolved model specification, or `None` if no explicit value or relevant environment variables are present.
    """
    if model_spec is not None:
        return model_spec
    config_text = os.environ.get("PORTEX_CANDIDATE_CONFIG")
    if config_text:
        return json.loads(config_text)
    return os.environ.get("PORTEX_CANDIDATE_MODEL")


def _get_candidate_provider(model_spec: ModelSpec | None = None) -> Provider | None:
    """
    Resolve and return a Provider for a candidate model specification.
    
    If no `model_spec` is supplied, the function attempts to resolve a candidate model spec from environment/configuration. Returns `None` when no model spec can be determined.
    
    Parameters:
        model_spec (ModelSpec | None): Optional explicit model specification to use; if omitted, the spec is resolved from environment/configuration.
    
    Returns:
        Provider | None: A Provider for the resolved candidate model spec, or `None` if no spec is available.
    """
    model_spec = _candidate_provider_spec(model_spec)
    if not model_spec:
        return None
    return get_provider(model_spec)


def _extract_prompt_from_messages(messages: list[ChatMessage]) -> str:
    """
    Aggregate textual content from a sequence of ChatMessage objects into a single prompt string.
    
    This collects each message's content: if content is a string it's appended; if content is a list, items with a `text` attribute are appended. Parts are joined with newline characters.
    
    Parameters:
        messages (list[ChatMessage]): Sequence of chat messages to extract text from.
    
    Returns:
        str: The combined prompt composed of extracted text segments separated by newlines.
    """
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
    """
    Serialize a sequence of ChatMessage objects into a provider-compatible list of message dictionaries.
    
    Each input message with role "system", "user", or "assistant" is preserved; its content is serialized as either:
    - a plain string -> {"role": role, "content": <string>}
    - a list of content items -> {"role": role, "content": [<serialized parts>]}
    
    Serialized parts may be:
    - text entries: {"type": "text", "text": <text>}
    - image entries: {"type": "image", "image": <url_or_path>, "detail": <detail_or_"auto">}
    - document entries: converted to a text entry describing the document and its inferred MIME type, e.g. {"type": "text", "text": "[document: NAME (MIME)]"}
    
    Parameters:
        messages (list[ChatMessage]): Sequence of chat messages to serialize. Each message is expected to have a `role` attribute and a `content` attribute that is either a string or a list of content items.
    
    Returns:
        list[dict[str, Any]]: Provider-compatible list of message dictionaries.
    """
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
    """
    Extract token-level logprob information from a provider response into a Logprobs object.
    
    Parses a provider response dict and, if it contains a non-empty "choices" list whose first choice
    includes a "logprobs" dict with a "content" list, constructs a Logprobs object whose entries
    represent each token's `token`, `logprob`, optional `bytes`, and optional `top_logprobs` list.
    Returns None when the input does not have the expected structure.
    
    Returns:
        Logprobs | None: A Logprobs object built from the first choice's content entries, or `None`
        if `raw` is not a dict or is missing/invalid `choices`, `logprobs`, or `content`.
    """
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
    """
    Builds a ModelOutput representing a provider response.
    
    Parameters:
        provider (Provider): The provider that produced the response; its provider_id and model_name are combined into the output model identifier.
        text (str): The response text content to place in the output.
        usage (dict[str, int] | None): Optional token usage dictionary. Expected keys: "input_tokens", "output_tokens", "total_tokens". Missing or falsy values are treated as 0.
        raw (Any): The raw provider response used to extract token-level logprobs.
    
    Returns:
        ModelOutput: A ModelOutput whose `model` is "{provider_id}:{model_name}", whose content is `text`, whose `usage` (if provided) is populated with integer token counts defaulting to 0, and whose first choice's `logprobs` (if any) are set from the extracted provider logprobs.
    """
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
    """
    Create a Solver that generates candidate-role model outputs, either via the builtin model system or an external provider.
    
    When invoked, the returned Solver produces generation for the "candidate" role and appends the produced message to the TaskState's messages. In provider mode, generation is performed through the portex_eval provider abstraction; in non-provider mode, it delegates to the normal generate flow and ensures the active model's role is set to "candidate".
    
    Parameters:
        model_string: Optional model identifier used to select a provider when `use_provider` is True. If not provided in provider mode, the function will attempt to read configuration from environment variables.
        tool_calls: Controls tool-call handling mode for the non-provider generate flow; one of "loop", "single", or "none".
        use_provider: If True, use an external provider for generation instead of the builtin model system.
        **kwargs: Additional keyword arguments forwarded to the underlying generate or provider call.
    
    Returns:
        A Solver function that performs candidate-role generation and updates the TaskState with the generated output.
    
    Raises:
        ValueError: If `use_provider` is True but no provider can be resolved from `model_string` or environment configuration.
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
        """
        Handle candidate generation for a task by either calling an external provider or delegating to the local generator.
        
        If a provider is available and configured to be used, requests a candidate from the provider, stores the resulting ModelOutput on state.output, appends the produced message to state.messages, and returns the updated state. Otherwise, ensures the active local model has the "candidate" role and delegates generation to the provided `generate` callable with the configured tool calling behavior.
        
        Parameters:
            state (TaskState): The current task state to read from and modify (messages and output).
            generate (Generate): Fallback generation callable used when not using an external provider.
        
        Returns:
            TaskState: The updated task state containing the generated output and appended message.
        """
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
    """
    Create a Solver that generates outputs using the specified provider model spec.
    
    The returned Solver calls the provider's async generate with the current chat messages, converts the provider response into a ModelOutput assigned to state.output, and appends the produced message to state.messages.
    
    Parameters:
        model_spec (ModelSpec): Provider model identifier or configuration to obtain a provider.
        max_tokens (int | None): Maximum number of tokens the provider should generate.
        temperature (float | None): Sampling temperature for generation.
        **kwargs: Additional keyword arguments forwarded to the provider's generate call.
    
    Returns:
        Solver: A function that accepts (state: TaskState, generate: Generate) and returns an updated TaskState with the provider-generated output.
    """
    provider = get_provider(model_spec)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        """
        Send the current conversation to the configured provider for generation, store the provider's ModelOutput on the state, and append the produced message to state.messages.
        
        Parameters:
            state (TaskState): The task state containing the conversation messages; this function will set state.output and append the provider-produced message to state.messages.
            generate (Generate): Unused in this provider-backed solver but kept for signature compatibility.
        
        Returns:
            TaskState: The updated task state with output populated and the new message appended.
        """
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
