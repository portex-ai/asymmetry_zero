"""Portex scorer with multi-judge support and provider abstraction."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, cast

from inspect_ai._util._async import tg_collect
from inspect_ai.model import Model, get_model
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    model_graded_qa,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from portex_eval.grading import (
    GRADING_INSTRUCTIONS,
    GRADING_TEMPLATE,
    criterion_grader_type,
    criterion_prompt,
    extract_final_answer,
    grade_criterion_exact_match,
    grade_criterion_with_providers as _grade_criterion_with_shared_providers,
    load_answer_key,
)
from portex_eval.providers import ModelSpec, Provider, get_provider

logger = logging.getLogger(__name__)


def _score_to_bool(score: Score) -> bool:
    """
    Determine whether a Score represents a correct result.
    
    Returns:
        True if the score's value equals `CORRECT` (case-insensitive), False otherwise.
    """
    return bool(str(score.value).upper() == CORRECT)


_load_answer_key = load_answer_key
_criterion_prompt = criterion_prompt
_criterion_grader_type = criterion_grader_type
_extract_final_answer = extract_final_answer
_grade_criterion_exact_match = grade_criterion_exact_match


def _ensure_model_role(model: Model, role: str) -> Model:
    """
    Ensure the given Model has the specified role and return the model.
    
    Parameters:
        model (Model): The model whose role should be checked and updated if different.
        role (str): The role to assign to the model when its current role does not match.
    
    Returns:
        Model: The same model instance, with its role set to `role` if it was different.
    """
    if model.role != role:
        model._set_role(role)
    return model


def _resolve_judge_models(
    model: Sequence[str | Model] | str | Model | None,
) -> list[Model]:
    """
    Convert various judge model specifications into a list of Inspect Model instances with role set to "grader".
    
    Parameters:
        model: A model specification which may be:
            - None => treated as no judges
            - a single Model instance
            - a single model name string
            - a sequence (list or tuple) of Model instances and/or model name strings
    
    Returns:
        list[Model]: A list of Inspect Model instances with their role ensured to "grader". Empty list if `model` is None or not a recognized form.
    """
    if model is None:
        return []
    if isinstance(model, list | tuple):
        models: list[Model] = []
        for item in model:
            if isinstance(item, Model):
                models.append(_ensure_model_role(item, "grader"))
            else:
                if isinstance(item, str):
                    models.append(get_model(item, role="grader"))
        return models
    if isinstance(model, Model):
        return [_ensure_model_role(model, "grader")]
    if isinstance(model, str):
        return [get_model(model, role="grader")]
    return []


def _resolve_judge_providers(model_specs: list[ModelSpec]) -> dict[str, Provider]:
    """
    Convert a list of ModelSpec into Provider instances keyed by "provider_id:model_name".
    
    Parameters:
        model_specs (list[ModelSpec]): List of model specifications to resolve to providers.
    
    Returns:
        dict[str, Provider]: Mapping from "provider_id:model_name" to the resolved Provider instance.
    """
    providers: dict[str, Provider] = {}
    for model_spec in model_specs:
        provider = get_provider(model_spec)
        providers[f"{provider.provider_id}:{provider.model_name}"] = provider
    return providers




@scorer(metrics=[accuracy(), stderr()])
def portex_scorer(
    model: list[ModelSpec] | ModelSpec | list[Model] | str | Model | None = None,
    answers_path: str | None = None,
    use_provider: bool = False,
) -> Scorer:
    """Portex scorer with multi-judge support.

    This scorer can operate in two modes:
    1. Standard Inspect mode (use_provider=False): Uses Inspect's model_graded_qa
    2. Provider mode (use_provider=True): Uses portex_eval.providers with rate
       limit handling and exponential backoff

    Args:
        model: Judge model(s). Can be:
            - A single model string or config object
            - A list of model strings/config objects for multi-judge scoring
            - An Inspect Model instance (if use_provider=False)
        answers_path: Path to answers.json file.
        use_provider: If True, use portex_eval.providers for judge calls.

    Returns:
        Scorer function.
    """
    answers_path = answers_path or str(Path(__file__).with_name("answers.json"))
    answer_key = _load_answer_key(answers_path)

    judge_providers: dict[str, Provider] = {}
    judge_models: list[Model] = []
    judge_scorers: dict[str, Scorer] = {}
    criterion_scorer: Scorer | None = None

    if use_provider:
        if model is None:
            raise ValueError("model is required when use_provider=True")
        model_list = [model] if not isinstance(model, list | tuple) else list(model)
        if any(isinstance(item, Model) for item in model_list):
            raise ValueError(
                "Inspect Model instances are not supported when use_provider=True. "
                "Use provider model strings or config objects instead."
            )
        judge_providers = _resolve_judge_providers(cast(list[ModelSpec], model_list))
    else:
        if isinstance(model, list | tuple):
            judge_models = _resolve_judge_models(model)
        elif isinstance(model, str | Model) or model is None:
            judge_models = _resolve_judge_models(model)
        else:
            judge_models = []
        if judge_models:
            if len(judge_models) == 1:
                criterion_model: list[str | Model] | str | Model | None = judge_models[0]
            else:
                criterion_model = cast(list[str | Model], judge_models)
        else:
            criterion_model = None
        criterion_scorer = model_graded_qa(
            model=criterion_model, template=GRADING_TEMPLATE, instructions=GRADING_INSTRUCTIONS
        )
        judge_scorers = {
            str(judge_model): model_graded_qa(
                model=judge_model,
                template=GRADING_TEMPLATE,
                instructions=GRADING_INSTRUCTIONS,
            )
            for judge_model in judge_models
        }

    async def score(state: TaskState, target: Target) -> Score:
        """
        Grade a single TaskState against the answer key and return a Score summarizing pass/fail and per-criterion results.
        
        Loads the answer entry for the task, extracts criteria and the submission, grades each criterion using either provider-based graders or Inspect-based graders (depending on scorer configuration), aggregates awarded weights, and marks the task as passed when the total awarded score meets or exceeds the criterion pass threshold.
        
        Parameters:
            state (TaskState): The task state containing messages, sample_id, and output to grade.
            target (Target): The target used for scoring (passed through to underlying scorers).
        
        Returns:
            Score: A Score whose value is CORRECT when the aggregated awarded score >= passThreshold, otherwise INCORRECT. The Score includes the original answer (if present), an explanation with the total score and threshold, and metadata with task_id, total_score, pass_threshold, and per-criterion grading details.
        
        Raises:
            ValueError: If an expected criterion scorer or judge scorer returns None or is missing when required.
        """
        task_id = str(state.sample_id)
        answer_entry = answer_key.get(task_id)
        if answer_entry is None:
            return Score(
                value=INCORRECT,
                explanation=f"No answer entry found for task_id '{task_id}'",
                metadata={"task_id": task_id, "answers_path": answers_path},
            )

        criteria: Iterable[dict[str, Any]] = answer_entry.get("criteria", [])
        pass_threshold = float(answer_entry.get("passThreshold", 70))

        question = ""
        for msg in state.messages:
            if hasattr(msg, "content"):
                if isinstance(msg.content, str):
                    question = msg.content
                    break
                elif isinstance(msg.content, list):
                    for item in msg.content:
                        if hasattr(item, "text"):
                            question = item.text
                            break
                    if question:
                        break

        submission = state.output.completion if state.output else ""

        async def grade_criterion_with_providers(criterion: dict[str, Any]) -> dict[str, Any]:
            """
            Grade a single criterion using provider-based judges.
            
            Parameters:
                criterion (dict[str, Any]): Criterion entry from the answer key describing id, name, weight, prompt, and grader configuration.
            
            Returns:
                dict[str, Any]: Grading result for the criterion, including keys such as `criterion_id`, `name`, `grader_type`, `weight`, `grade`, `passed`, `awarded`, `explanation`, and `judges`.
            """
            return await _grade_criterion_with_shared_providers(
                criterion,
                question=question,
                submission=submission,
                judge_providers=judge_providers,
            )

        async def grade_criterion_with_inspect(criterion: dict[str, Any]) -> dict[str, Any]:
            """
            Evaluate a single grading criterion using either an exact-match grader or one or more Inspect judge scorers.
            
            Parameters:
                criterion (dict): Criterion specification. Expected keys include:
                    - "id": criterion identifier
                    - "name": criterion name
                    - fields used by the criterion prompt generator (consumed by _criterion_prompt)
                    - "weight" (optional): numeric weight for the criterion
                    - any grader selection metadata used by _criterion_grader_type
            
            Returns:
                dict: A result object with the following keys:
                    - "criterion_id": the criterion's id
                    - "name": the criterion's name
                    - "prompt": the prompt text used for grading
                    - "grader_type": string indicating the grader type (e.g., "llm-judge")
                    - "weight": numeric weight awarded to the criterion
                    - "grade": grader output value or label
                    - "passed": `true` if the criterion is considered passed, `false` otherwise
                    - "awarded": numeric points awarded (weight if passed, otherwise 0.0)
                    - "explanation": human-readable explanation of the grading outcome
                    - "judges": list of per-judge result objects when multiple judges were used
            """
            grader_type = _criterion_grader_type(criterion)
            if grader_type == "ExactMatch":
                return _grade_criterion_exact_match(criterion, submission)
            prompt = _criterion_prompt(criterion)
            weight = float(criterion.get("weight", 0))

            if judge_scorers:
                judge_tasks: list[Callable[[], Awaitable[dict[str, Any]]]] = []
                for name, judge_scorer in judge_scorers.items():

                    def _task(
                        scorer: Scorer = judge_scorer,
                        model_name: str = name,
                    ) -> Awaitable[dict[str, Any]]:
                        return _grade_judge_inspect(scorer, model_name, state, prompt, weight)

                    judge_tasks.append(_task)
                judge_results: list[dict[str, Any]] = await tg_collect(judge_tasks)
                passed_count = sum(1 for r in judge_results if r["passed"])
                majority_passed = passed_count > len(judge_results) / 2
                awarded = weight if majority_passed else 0.0
                grade_value = CORRECT if majority_passed else INCORRECT
                explanation = f"Majority vote: {passed_count}/{len(judge_results)} judges passed"
            else:
                if criterion_scorer is None:
                    raise ValueError(
                        "criterion_scorer is required when no judge_scorers are defined"
                    )
                result = await criterion_scorer(state, Target(prompt))
                if result is None:
                    raise ValueError("criterion_scorer returned None")
                passed = _score_to_bool(result)
                awarded = weight if passed else 0.0
                grade_value = str(result.value)
                majority_passed = passed
                explanation = str(result.explanation)
                judge_results = []

            return {
                "criterion_id": criterion.get("id"),
                "name": criterion.get("name"),
                "prompt": prompt,
                "grader_type": "llm-judge",
                "weight": weight,
                "grade": grade_value,
                "passed": majority_passed,
                "awarded": awarded,
                "explanation": explanation,
                "judges": judge_results,
            }

        async def _grade_judge_inspect(
            judge_scorer: Scorer,
            model_name: str,
            state: TaskState,
            prompt: str,
            weight: float,
        ) -> dict[str, Any]:
            judge_result = await judge_scorer(state, Target(prompt))
            if judge_result is None:
                raise ValueError("judge_scorer returned None")
            judge_passed = _score_to_bool(judge_result)
            return {
                "model": model_name,
                "grade": str(judge_result.value),
                "passed": judge_passed,
                "awarded": weight if judge_passed else 0.0,
                "explanation": judge_result.explanation,
            }

        grade_criterion = (
            grade_criterion_with_providers if use_provider else grade_criterion_with_inspect
        )

        if criteria:
            grade_tasks: list[Callable[[], Awaitable[dict[str, Any]]]] = []
            for criterion in criteria:

                def _task(criterion: dict[str, Any] = criterion) -> Awaitable[dict[str, Any]]:
                    return grade_criterion(criterion)

                grade_tasks.append(_task)
            results: list[dict[str, Any]] = await tg_collect(grade_tasks)
        else:
            results = []

        total_score = sum(result["awarded"] for result in results)
        passed = total_score >= pass_threshold

        return Score(
            value=CORRECT if passed else INCORRECT,
            answer=state.output.completion if state.output else None,
            explanation=f"Total score {total_score} (passThreshold {pass_threshold})",
            metadata={
                "task_id": task_id,
                "total_score": total_score,
                "pass_threshold": pass_threshold,
                "criteria": results,
            },
        )

    return score


@scorer(metrics=[accuracy(), stderr()])
def provider_scorer(
    judges: list[ModelSpec],
    answers_path: str | None = None,
) -> Scorer:
    """
    Create a Scorer that grades using provider-based judges.
    
    Parameters:
        judges (list[ModelSpec]): List of provider model specifications to use as judges.
        answers_path (str | None): Path to the answers.json file; if None the default answers file is used.
    
    Returns:
        Scorer: A scorer function configured to run multi-judge grading via the provider abstraction.
    """
    return portex_scorer(model=judges, answers_path=answers_path, use_provider=True)
