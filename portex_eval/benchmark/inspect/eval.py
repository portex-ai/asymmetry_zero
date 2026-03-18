"""Inspect AI task definitions for portex_eval.

This module provides task factories that can be used with Inspect AI's
evaluation runner. Tasks can use either Inspect's built-in model handling
or the portex_eval provider abstraction with rate limit handling.
"""

from __future__ import annotations

import json
import os
from typing import Any

from inspect_ai import Task, task
from inspect_ai.solver import system_message

from portex_eval.benchmark.inspect.dataset import dataset_generator
from portex_eval.benchmark.inspect.scorer import portex_scorer, provider_scorer
from portex_eval.benchmark.inspect.solver import candidate_generate, provider_generate
from portex_eval.providers import ModelSpec

SYSTEM_MESSAGE = """
Complete the task to the best of your ability.

After your reasoning, state your final answer clearly on a single line that starts
with Answer: so it can be parsed automatically. Example:
Answer: The capital of France is Paris.
"""

DEFAULT_JUDGE_MODELS_INSPECT = [
    "openrouter/meta-llama/llama-3-8b-instruct",
    "openrouter/qwen/qwen3-vl-8b-instruct",
    "openrouter/openai/gpt-oss-20b",
]

DEFAULT_JUDGE_MODELS_PROVIDER = [
    "openrouter:meta-llama/llama-3-8b-instruct",
    "openrouter:qwen/qwen3-vl-8b-instruct",
    "openrouter:openai/gpt-oss-20b",
]

DATA_DIR = os.getenv("PORTEX_EVAL_DATA_DIR", "./bundle")


def _parse_env_list(env_var: str, default: list[str]) -> list[str]:
    """
    Parse an environment variable containing comma-separated values into a list of trimmed strings.
    
    Parameters:
    	env_var (str): Name of the environment variable to read.
    	default (list[str]): Value to return if the environment variable is not set or empty.
    
    Returns:
    	list[str]: List of non-empty, trimmed values from the environment variable, or `default` if the variable is not set or empty.
    """
    value = os.getenv(env_var)
    if not value:
        return default
    return [m.strip() for m in value.split(",") if m.strip()]


def _parse_env_json(env_var: str) -> Any | None:
    """
    Parse a JSON value from an environment variable.
    
    Parameters:
        env_var (str): Name of the environment variable to read.
    
    Returns:
        Any | None: The Python value obtained by JSON-decoding the environment variable's contents, or `None` if the environment variable is not set or is empty.
    """
    value = os.getenv(env_var)
    if not value:
        return None
    return json.loads(value)


def _candidate_provider_spec() -> ModelSpec | None:
    """
    Resolve the candidate model specification for provider-based evaluation from environment variables.
    
    Checks PORTEX_CANDIDATE_CONFIG for a JSON configuration and returns it if present; otherwise falls back to PORTEX_CANDIDATE_MODEL. 
    
    Returns:
        ModelSpec | None: The candidate model specification parsed from `PORTEX_CANDIDATE_CONFIG`, the string value of `PORTEX_CANDIDATE_MODEL` if set, or `None` if neither is provided.
    """
    config = _parse_env_json("PORTEX_CANDIDATE_CONFIG")
    if config is not None:
        return config
    return os.getenv("PORTEX_CANDIDATE_MODEL")


def _judge_provider_specs(default: list[str]) -> list[ModelSpec]:
    """
    Resolve the judge model specifications for provider-based evaluation.
    
    If the environment variable PORTEX_JUDGE_CONFIGS contains a non-empty JSON list, that list is returned. Otherwise, the function reads PORTEX_JUDGE_MODELS and returns its parsed entries; if PORTEX_JUDGE_MODELS is unset, the provided `default` is returned.
    
    Parameters:
        default (list[str]): Fallback list of judge model identifiers used when PORTEX_JUDGE_MODELS is not set.
    
    Returns:
        list[ModelSpec]: A list of judge model specifications to use for scoring.
    """
    configs = _parse_env_json("PORTEX_JUDGE_CONFIGS")
    if isinstance(configs, list) and configs:
        return configs
    return _parse_env_list("PORTEX_JUDGE_MODELS", default)


def _candidate_generation_options() -> dict[str, Any]:
    """
    Builds generation options for candidate providers from environment variables.
    
    Reads PORTEX_LOGPROBS and PORTEX_TOP_LOGPROBS. If PORTEX_LOGPROBS is set to "1", "true", or "yes" (case-insensitive), the returned dict includes "logprobs": True. If PORTEX_TOP_LOGPROBS is set, it is parsed as an integer and added as "top_logprobs"; setting "top_logprobs" also enables "logprobs".
    
    Returns:
        options (dict[str, Any]): Dictionary with optional keys:
            - "logprobs" (bool): True when log probabilities are requested.
            - "top_logprobs" (int): Number of top logprob buckets when specified.
    """
    options: dict[str, Any] = {}
    if os.getenv("PORTEX_LOGPROBS", "").lower() in {"1", "true", "yes"}:
        options["logprobs"] = True
    top_logprobs = os.getenv("PORTEX_TOP_LOGPROBS")
    if top_logprobs:
        options["top_logprobs"] = int(top_logprobs)
        options["logprobs"] = True
    return options


@task
def portex_qa_eval() -> Task:
    """Standard Portex QA evaluation task using Inspect's model handling.

    This task uses Inspect's built-in model system for both candidate
    generation and judge scoring. Configure via environment variables:

    - PORTEX_EVAL_DATA_DIR: Path to bundle directory (default: ./bundle)
    - PORTEX_JUDGE_MODELS: Comma-separated list of judge models

    Returns:
        Inspect Task instance.
    """
    data_dir = os.getenv("PORTEX_EVAL_DATA_DIR", DATA_DIR) or DATA_DIR
    judge_models = _parse_env_list("PORTEX_JUDGE_MODELS", DEFAULT_JUDGE_MODELS_INSPECT)

    return Task(
        dataset=dataset_generator(str(data_dir + "/tasks.json")),
        solver=[system_message(SYSTEM_MESSAGE), candidate_generate()],
        scorer=portex_scorer(
            model=judge_models,
            answers_path=str(data_dir + "/answers.json"),
        ),
    )


@task
def portex_qa_eval_with_providers() -> Task:
    """
    Portex QA evaluation task configured to use provider-based candidate generation and judge scoring.
    
    Builds and returns a Task that uses portex_eval.providers for generation (with optional logprobs/top_logprobs) and provider-based scoring. The candidate model must be provided via environment configuration (e.g., PORTEX_CANDIDATE_CONFIG or PORTEX_CANDIDATE_MODEL); judge models are resolved from provider configs or PORTEX_JUDGE_MODELS with sensible defaults.
    
    Returns:
        Task: An Inspect Task configured to generate candidate answers with providers and score them using provider-based judges.
    
    Raises:
        ValueError: If no candidate model specification is available from the environment (candidate model is required for provider mode).
    """
    data_dir = os.getenv("PORTEX_EVAL_DATA_DIR", DATA_DIR) or DATA_DIR
    candidate_model = _candidate_provider_spec()
    judge_models = _judge_provider_specs(DEFAULT_JUDGE_MODELS_PROVIDER)

    if not candidate_model:
        raise ValueError(
            "PORTEX_CANDIDATE_MODEL environment variable is required for provider mode. "
            "Example: PORTEX_CANDIDATE_MODEL=openrouter:google/gemini-2.5-flash"
        )

    return Task(
        dataset=dataset_generator(str(data_dir + "/tasks.json")),
        solver=[
            system_message(SYSTEM_MESSAGE),
            provider_generate(candidate_model, **_candidate_generation_options()),
        ],
        scorer=provider_scorer(
            judges=judge_models,
            answers_path=str(data_dir + "/answers.json"),
        ),
    )


def create_eval_task(
    data_dir: str | None = None,
        candidate_model: ModelSpec | None = None,
        judge_models: list[ModelSpec] | None = None,
    use_providers: bool = False,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> Task:
    """Create a Portex evaluation task programmatically.

    This factory function allows creating evaluation tasks with explicit
    configuration rather than relying on environment variables.

    Args:
        data_dir: Path to bundle directory containing tasks.json and answers.json.
        candidate_model: Candidate model string or config object.
            Required if use_providers=True.
        judge_models: List of judge model strings or config objects.
        use_providers: If True, use portex_eval.providers for generation and scoring.
            This enables rate limit handling with exponential backoff.
        system_prompt: Custom system prompt. Defaults to SYSTEM_MESSAGE.
        **kwargs: Additional arguments (reserved for future use).

    Returns:
        Configured Inspect Task instance.

    Example:
        >>> task = create_eval_task(
        ...     data_dir="./my_bundle",
        ...     candidate_model="openrouter:google/gemini-2.5-flash",
        ...     judge_models=[
        ...         "openrouter:anthropic/claude-3.5-sonnet",
        ...         "openrouter:openai/gpt-4o",
        ...     ],
        ...     use_providers=True,
        ... )
    """
    data_dir = data_dir or os.getenv("PORTEX_EVAL_DATA_DIR", DATA_DIR) or DATA_DIR
    if use_providers:
        judge_models = judge_models or _judge_provider_specs(DEFAULT_JUDGE_MODELS_PROVIDER)
    else:
        judge_models = judge_models or _parse_env_list(
            "PORTEX_JUDGE_MODELS", DEFAULT_JUDGE_MODELS_INSPECT
        )
    system_prompt = system_prompt or SYSTEM_MESSAGE

    if use_providers:
        candidate_model = candidate_model or _candidate_provider_spec()
        if not candidate_model:
            raise ValueError(
                "candidate_model is required when use_providers=True. "
                "Example: candidate_model='openrouter:google/gemini-2.5-flash'"
            )
        return Task(
            dataset=dataset_generator(str(data_dir + "/tasks.json")),
            solver=[
                system_message(system_prompt),
                provider_generate(candidate_model, **_candidate_generation_options()),
            ],
            scorer=provider_scorer(
                judges=judge_models,
                answers_path=str(data_dir + "/answers.json"),
            ),
        )
    else:
        return Task(
            dataset=dataset_generator(str(data_dir + "/tasks.json")),
            solver=[
                system_message(system_prompt),
                candidate_generate(model_string=candidate_model, use_provider=False),
            ],
            scorer=portex_scorer(
                model=judge_models,
                answers_path=str(data_dir + "/answers.json"),
                use_provider=False,
            ),
        )
