"""Inspect AI task definitions for portex_eval.

This module provides task factories that can be used with Inspect AI's
evaluation runner. Tasks can use either Inspect's built-in model handling
or the portex_eval provider abstraction with rate limit handling.
"""

from __future__ import annotations

import os
from typing import Any

from inspect_ai import Task, task
from inspect_ai.solver import system_message

from portex_eval.benchmark.inspect.dataset import dataset_generator
from portex_eval.benchmark.inspect.scorer import portex_scorer, provider_scorer
from portex_eval.benchmark.inspect.solver import candidate_generate, provider_generate

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
    """Parse a comma-separated environment variable into a list."""
    value = os.getenv(env_var)
    if not value:
        return default
    return [m.strip() for m in value.split(",") if m.strip()]


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
    """Portex QA evaluation task using provider abstraction.

    This task uses portex_eval.providers for both candidate generation
    and judge scoring, which includes rate limit handling with exponential
    backoff. Configure via environment variables:

    - PORTEX_EVAL_DATA_DIR: Path to bundle directory (default: ./bundle)
    - PORTEX_CANDIDATE_MODEL: Candidate model string (e.g., openrouter:google/gemini-2.5-flash)
    - PORTEX_JUDGE_MODELS: Comma-separated list of judge model strings

    Returns:
        Inspect Task instance.
    """
    data_dir = os.getenv("PORTEX_EVAL_DATA_DIR", DATA_DIR) or DATA_DIR
    candidate_model = os.getenv("PORTEX_CANDIDATE_MODEL")
    judge_models = _parse_env_list("PORTEX_JUDGE_MODELS", DEFAULT_JUDGE_MODELS_PROVIDER)

    if not candidate_model:
        raise ValueError(
            "PORTEX_CANDIDATE_MODEL environment variable is required for provider mode. "
            "Example: PORTEX_CANDIDATE_MODEL=openrouter:google/gemini-2.5-flash"
        )

    return Task(
        dataset=dataset_generator(str(data_dir + "/tasks.json")),
        solver=[
            system_message(SYSTEM_MESSAGE),
            provider_generate(candidate_model),
        ],
        scorer=provider_scorer(
            judges=judge_models,
            answers_path=str(data_dir + "/answers.json"),
        ),
    )


def create_eval_task(
    data_dir: str | None = None,
    candidate_model: str | None = None,
    judge_models: list[str] | None = None,
    use_providers: bool = False,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> Task:
    """Create a Portex evaluation task programmatically.

    This factory function allows creating evaluation tasks with explicit
    configuration rather than relying on environment variables.

    Args:
        data_dir: Path to bundle directory containing tasks.json and answers.json.
        candidate_model: Candidate model string (e.g., 'openrouter:google/gemini-2.5-flash').
            Required if use_providers=True.
        judge_models: List of judge model strings. Defaults to DEFAULT_JUDGE_MODELS.
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
        judge_models = judge_models or _parse_env_list(
            "PORTEX_JUDGE_MODELS", DEFAULT_JUDGE_MODELS_PROVIDER
        )
    else:
        judge_models = judge_models or _parse_env_list(
            "PORTEX_JUDGE_MODELS", DEFAULT_JUDGE_MODELS_INSPECT
        )
    system_prompt = system_prompt or SYSTEM_MESSAGE

    if use_providers:
        candidate_model = candidate_model or os.getenv("PORTEX_CANDIDATE_MODEL")
        if not candidate_model:
            raise ValueError(
                "candidate_model is required when use_providers=True. "
                "Example: candidate_model='openrouter:google/gemini-2.5-flash'"
            )
        return Task(
            dataset=dataset_generator(str(data_dir + "/tasks.json")),
            solver=[
                system_message(system_prompt),
                provider_generate(candidate_model),
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
