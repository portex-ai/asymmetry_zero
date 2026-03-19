"""Custom Harbor agents for Portex-specific task execution."""

from __future__ import annotations

import json
import math
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import BaseLLM, LLMResponse
from harbor.llms.chat import Chat
from harbor.models.agent.context import AgentContext
from harbor.models.metric import UsageInfo
from harbor.models.trajectories import Step

from portex_eval.providers import get_provider

IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}
TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".tsv",
    ".xml",
    ".html",
    ".py",
    ".js",
    ".ts",
    ".sql",
}
MAX_TEXT_REFERENCE_BYTES = 200_000
FALLBACK_CONTEXT_LIMIT = 128_000
FALLBACK_OUTPUT_LIMIT = 8_192
REFERENCE_FILE_PATH_RE = re.compile(r"Reference file path:\s*`([^`]+)`")
ANSWER_PATH_RE = re.compile(r"Write your complete response .* to `([^`]+)`\.", re.DOTALL)
NO_REFERENCE_SENTINELS = {"(none)", "none", "null", "n/a"}


def _provider_spec(
    *,
    model_name: str,
    provider: str | None,
    api_base: str | None,
    api_key: str | None,
    api_key_env: str | None,
    headers: dict[str, str] | None,
    options: dict[str, Any] | None,
) -> dict[str, Any]:
    resolved_provider = provider
    resolved_model = model_name

    if ":" in model_name:
        provider_prefix, resolved_model = model_name.split(":", 1)
        resolved_provider = provider_prefix
    elif "/" in model_name:
        provider_prefix, rest = model_name.split("/", 1)
        if provider_prefix in {
            "openrouter",
            "openai",
            "anthropic",
            "openai_compatible",
            "openai-compatible",
            "vllm",
            "custom",
            "hosted_vllm",
        }:
            resolved_provider = provider_prefix
            resolved_model = rest

    if resolved_provider in {None, ""}:
        resolved_provider = "custom" if api_base else "openrouter"
    if resolved_provider == "hosted_vllm":
        resolved_provider = "vllm"

    config: dict[str, Any] = {
        "provider": resolved_provider,
        "model": resolved_model,
    }
    if api_base:
        config["base_url"] = api_base
    if api_key:
        config["api_key"] = api_key
    if api_key_env:
        config["api_key_env"] = api_key_env
    if headers:
        config["headers"] = dict(headers)
    if options:
        config["options"] = dict(options)
    return config


def _response_reasoning_content(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    message = choice.get("message")
    if not isinstance(message, dict):
        return None
    reasoning = message.get("reasoning_content")
    return reasoning if isinstance(reasoning, str) else None


def _response_logprobs(raw: Any) -> list[float] | None:
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
    values: list[float] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        logprob = item.get("logprob")
        if isinstance(logprob, (int, float)):
            values.append(float(logprob))
    return values or None


def _response_completion_token_ids(raw: Any) -> list[int] | None:
    if not isinstance(raw, dict):
        return None
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None

    direct_ids = choice.get("token_ids")
    if isinstance(direct_ids, list) and all(isinstance(item, int) for item in direct_ids):
        return list(direct_ids)

    provider_fields = choice.get("provider_specific_fields")
    if isinstance(provider_fields, dict):
        nested_ids = provider_fields.get("token_ids")
        if isinstance(nested_ids, list) and all(isinstance(item, int) for item in nested_ids):
            return list(nested_ids)
    return None


class PortexProviderLLM(BaseLLM):
    """Harbor LLM adapter backed by Portex provider implementations."""

    def __init__(
        self,
        *,
        model_name: str,
        provider: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        headers: dict[str, str] | None = None,
        options: dict[str, Any] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        model_info: dict[str, Any] | None = None,
        request_timeout_sec: float = 120.0,
    ) -> None:
        super().__init__()
        self._provider = get_provider(
            _provider_spec(
                model_name=model_name,
                provider=provider,
                api_base=api_base,
                api_key=api_key,
                api_key_env=api_key_env,
                headers=headers,
                options=options,
            ),
            timeout=request_timeout_sec,
        )
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._model_name = model_name
        self._model_info = dict(model_info or {})

    async def call(
        self,
        prompt: str,
        message_history: list[dict[str, Any]] = [],
        response_format: dict | None = None,
        logging_path: Path | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        del response_format
        messages = list(message_history)
        if prompt:
            messages.append({"role": "user", "content": prompt})

        response = await self._provider.agenerate(
            prompt,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            **kwargs,
        )
        if logging_path is not None:
            logging_path.write_text(json.dumps(response.raw, indent=2), encoding="utf-8")

        usage = response.usage or {}
        usage_info = UsageInfo(
            prompt_tokens=int(usage.get("input_tokens", 0) or 0),
            completion_tokens=int(usage.get("output_tokens", 0) or 0),
            cache_tokens=0,
            cost_usd=0.0,
        )
        return LLMResponse(
            content=response.text,
            reasoning_content=_response_reasoning_content(response.raw),
            usage=usage_info,
            prompt_token_ids=None,
            completion_token_ids=_response_completion_token_ids(response.raw),
            logprobs=_response_logprobs(response.raw),
        )

    def get_model_context_limit(self) -> int:
        max_input = self._model_info.get("max_input_tokens")
        max_tokens = self._model_info.get("max_tokens")
        for value in (max_input, max_tokens):
            if isinstance(value, (int, float)):
                return int(value)
        return FALLBACK_CONTEXT_LIMIT

    def get_model_output_limit(self) -> int | None:
        max_output = self._model_info.get("max_output_tokens")
        max_tokens = self._model_info.get("max_tokens")
        for value in (max_output, max_tokens):
            if isinstance(value, (int, float)):
                return int(value)
        return FALLBACK_OUTPUT_LIMIT


class PortexMultimodalChat(Chat):
    """Chat wrapper that sends a multimodal first turn but stores a text shadow."""

    def __init__(
        self,
        model: BaseLLM,
        *,
        first_user_content: list[dict[str, Any]] | None,
        first_user_shadow: str | None,
        interleaved_thinking: bool = False,
    ) -> None:
        super().__init__(model, interleaved_thinking=interleaved_thinking)
        self._first_user_content = first_user_content
        self._first_user_shadow = first_user_shadow
        self._used_first_user_content = False

    async def chat(
        self,
        prompt: str,
        logging_path: Path | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if self._used_first_user_content or not self._first_user_content or self._messages:
            return await super().chat(prompt, logging_path=logging_path, **kwargs)

        llm_response = await self._model.call(
            prompt="",
            message_history=[{"role": "user", "content": self._first_user_content}],
            logging_path=logging_path,
            **kwargs,
        )
        self._used_first_user_content = True

        usage = llm_response.usage
        if usage is not None:
            self._cumulative_input_tokens += usage.prompt_tokens
            self._cumulative_output_tokens += usage.completion_tokens
            self._cumulative_cache_tokens += usage.cache_tokens
            self._cumulative_cost += usage.cost_usd

        self._accumulate_rollout_details(llm_response)

        assistant_message = {"role": "assistant", "content": llm_response.content}
        if self._interleaved_thinking and llm_response.reasoning_content:
            assistant_message["reasoning_content"] = llm_response.reasoning_content

        self._messages.extend(
            [
                {"role": "user", "content": self._first_user_shadow or prompt},
                assistant_message,
            ]
        )
        return llm_response


class PortexMultimodalAgent(Terminus2):
    """A Terminus-2 style agent with first-turn multimodal Portex refs."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        provider: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        headers: dict[str, str] | None = None,
        options: dict[str, Any] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        model_info: dict[str, Any] | None = None,
        refs_dir: str = "/app/refs",
        request_timeout_sec: float = 120.0,
        *args,
        **kwargs,
    ) -> None:
        if model_name is None:
            raise ValueError("model_name is required for PortexMultimodalAgent")

        llm = PortexProviderLLM(
            model_name=model_name,
            provider=provider,
            api_base=api_base,
            api_key=api_key,
            api_key_env=api_key_env,
            headers=headers,
            options=options,
            temperature=temperature,
            max_tokens=max_tokens,
            model_info=model_info,
            request_timeout_sec=request_timeout_sec,
        )
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            temperature=temperature,
            model_info=model_info,
            llm=llm,
            *args,
            **kwargs,
        )
        self._refs_dir = refs_dir.rstrip("/")

    @staticmethod
    def name() -> str:
        return "portex-multimodal"

    def version(self) -> str | None:
        return "0.2.0"

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if self._session is None:
            raise RuntimeError("Session is not set")

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            terminal_state = self._limit_output_length(
                await self._session.get_incremental_output()
            )
            initial_prompt = self._prompt_template.format(
                instruction=instruction,
                terminal_state=terminal_state,
            )
            first_user_content, first_user_shadow, reference_meta = (
                await self._build_initial_user_message(
                    environment=environment,
                    instruction=instruction,
                    initial_prompt=initial_prompt,
                    temp_root=temp_root,
                )
            )

            self._chat = PortexMultimodalChat(
                self._llm,
                first_user_content=first_user_content,
                first_user_shadow=first_user_shadow,
                interleaved_thinking=self._interleaved_thinking,
            )
            self._context = context

            self._trajectory_steps.append(
                Step(
                    step_id=1,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="user",
                    message=first_user_shadow,
                )
            )

            actual_episodes = self._n_episodes
            try:
                actual_episodes = await self._run_agent_loop(
                    initial_prompt=initial_prompt,
                    chat=self._chat,
                    logging_dir=self.logs_dir,
                    original_instruction=instruction,
                )
            finally:
                context.rollout_details = (
                    self._chat.rollout_details + self._subagent_rollout_details
                )
                context.n_input_tokens = (
                    self._chat.total_input_tokens
                    + self._subagent_metrics.total_prompt_tokens
                )
                context.n_output_tokens = (
                    self._chat.total_output_tokens
                    + self._subagent_metrics.total_completion_tokens
                )
                context.n_cache_tokens = (
                    self._chat.total_cache_tokens
                    + self._subagent_metrics.total_cached_tokens
                )
                total_cost = self._chat.total_cost + self._subagent_metrics.total_cost_usd
                context.cost_usd = total_cost if total_cost > 0 else None
                context.metadata = {
                    "n_episodes": actual_episodes,
                    "api_request_times_msec": self._api_request_times,
                    "summarization_count": self._summarization_count,
                    "reference": reference_meta,
                }
                if self._store_all_messages:
                    context.metadata["all_messages"] = self._chat.messages
                self._dump_trajectory()

    def _count_total_tokens(self, chat: Chat) -> int:
        normalized_parts: list[str] = []
        for message in chat.messages:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                normalized_parts.append(content)
                continue
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    normalized_parts.append(str(item["text"]))
                elif item.get("type") == "image":
                    normalized_parts.append("[image attachment]")
        estimated = "".join(normalized_parts)
        return max(1, math.ceil(len(estimated) / 4))

    async def _build_initial_user_message(
        self,
        *,
        environment: BaseEnvironment,
        instruction: str,
        initial_prompt: str,
        temp_root: Path,
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None]:
        content: list[dict[str, Any]] = [{"type": "text", "text": initial_prompt}]
        shadow_parts = [initial_prompt]

        remote_ref_path = self._reference_path_from_instruction(instruction)
        if remote_ref_path is None:
            return content, initial_prompt, None

        reference_file = Path(remote_ref_path).name
        local_ref_path = temp_root / reference_file
        await environment.download_file(remote_ref_path, local_ref_path)

        suffix = local_ref_path.suffix.lower()
        reference_meta: dict[str, Any] = {
            "file": reference_file,
            "remote_path": remote_ref_path,
        }

        if suffix in IMAGE_SUFFIXES:
            note = f"Reference image path: `{remote_ref_path}`. The image is attached for this first model turn."
            content.append({"type": "text", "text": note})
            content.append({"type": "image", "image": str(local_ref_path), "detail": "high"})
            shadow_parts.append(note)
            shadow_parts.append(f"[Reference image was attached separately: {remote_ref_path}]")
            reference_meta["mode"] = "image"
            return content, "\n\n".join(shadow_parts), reference_meta

        file_bytes = local_ref_path.read_bytes()
        if suffix in TEXT_SUFFIXES or len(file_bytes) <= MAX_TEXT_REFERENCE_BYTES:
            text = file_bytes.decode("utf-8", errors="replace")
            if len(file_bytes) > MAX_TEXT_REFERENCE_BYTES:
                text = text[:MAX_TEXT_REFERENCE_BYTES]
                reference_meta["truncated"] = True
            note = f"Reference file `{remote_ref_path}` contents:\n\n{text}"
            content.append({"type": "text", "text": note})
            shadow_parts.append(note)
            reference_meta["mode"] = "text"
            return content, "\n\n".join(shadow_parts), reference_meta

        note = (
            f"The task includes a non-text reference file at `{remote_ref_path}`. "
            "It could not be attached as a supported multimodal input."
        )
        content.append({"type": "text", "text": note})
        shadow_parts.append(note)
        reference_meta["mode"] = "unsupported"
        return content, "\n\n".join(shadow_parts), reference_meta

    @staticmethod
    def _reference_path_from_instruction(instruction: str) -> str | None:
        match = REFERENCE_FILE_PATH_RE.search(instruction)
        if not match:
            return None
        reference_path = match.group(1).strip()
        if not reference_path:
            return None
        if reference_path.lower() in NO_REFERENCE_SENTINELS:
            return None
        return reference_path

    @staticmethod
    def _answer_path_from_instruction(instruction: str) -> str | None:
        match = ANSWER_PATH_RE.search(instruction)
        if not match:
            return None
        return match.group(1).strip() or None
