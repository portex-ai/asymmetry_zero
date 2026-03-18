"""Harness-neutral grading logic shared by Inspect and Harbor."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable
from typing import Any

from portex_eval.providers import ModelSpec, Provider, get_provider

CORRECT = "C"
INCORRECT = "I"

EXACT_MATCH_ANSWER_RE = re.compile(
    r"(?:^|\n)\s*(?:Final\s+)?Answer\s*:\s*(.+?)(?=\n|$)",
    re.IGNORECASE | re.DOTALL,
)

DEFAULT_AGENT_JUDGE_MODELS = [
    "openrouter:deepseek/deepseek-v3.2",
    "openrouter:z-ai/glm-5",
    "openrouter:openai/gpt-oss-120b",
    "openrouter:meta-llama/llama-3.3-70b-instruct",
    "openrouter:moonshotai/kimi-k2.5",
]

GRADING_TEMPLATE = """
You are assessing a submitted answer on a given task based on a criterion. Here is the data:

[BEGIN DATA]
***
[Task]: {question}
***
[Submission]: {answer}
***
[Criterion]: {criterion}
***
[END DATA]

Does the submission meet the criterion?

{instructions}
"""

GRADING_INSTRUCTIONS = """
After assessing the submitted answer, reply with 'GRADE: $LETTER' (without quotes)
where LETTER is one of CI. Please choose ONE option for the grade: either "C" for
correct answers, or "I" for incorrect answers.

For example, after reviewing a correct answer you might write 'GRADE: C' or after
reviewing an incorrect answer you might write 'GRADE: I'.

First, write out in a step by step manner your reasoning about the criterion to be
sure that your conclusion is correct. Avoid simply stating the correct answers at
the outset. Then, end with your answer formatted as 'GRADE: $LETTER' (without
quotes) where LETTER is one of CI.
"""


def load_answer_key(path: str) -> dict[str, dict[str, Any]]:
    """Load a task_id keyed answer map from answers.json."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    answer_key: dict[str, dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        task_id = item.get("task_id")
        if not isinstance(task_id, str):
            continue
        criteria = item.get("criteria")
        if not isinstance(criteria, list) or not criteria:
            item = {**item, "criteria": []}
        answer_key[task_id] = item
    return answer_key


def criterion_prompt(criterion: dict[str, Any]) -> str:
    return (
        criterion.get("semanticPrompt")
        or criterion.get("description")
        or criterion.get("name")
        or ""
    )


def criterion_grader_type(criterion: dict[str, Any]) -> str:
    return str(criterion.get("grader_type") or "llm-judge").strip()


def normalize_for_compare(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_final_answer(text: str) -> str:
    if not text or not text.strip():
        return ""
    stripped = text.strip()
    matches = list(EXACT_MATCH_ANSWER_RE.finditer(stripped))
    if matches:
        return matches[-1].group(1).strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return stripped


def grade_criterion_exact_match(criterion: dict[str, Any], submission: str) -> dict[str, Any]:
    prompt = criterion_prompt(criterion)
    weight = float(criterion.get("weight", 0))
    extracted = extract_final_answer(submission)
    reference = normalize_for_compare(prompt)
    predicted = normalize_for_compare(extracted)
    passed = bool(reference and reference in predicted)
    awarded = weight if passed else 0.0
    explanation = f"Reference: {prompt!r} | Extracted: {extracted!r} (include)"
    return {
        "criterion_id": criterion.get("id"),
        "name": criterion.get("name"),
        "prompt": prompt,
        "semanticPrompt": prompt,
        "grader_type": "ExactMatch",
        "weight": weight,
        "grade": CORRECT if passed else INCORRECT,
        "passed": passed,
        "awarded": awarded,
        "explanation": explanation,
        "judges": [
            {
                "model": "ExactMatch",
                "grade": "C" if passed else "I",
                "passed": passed,
                "awarded": awarded,
                "explanation": explanation,
            }
        ],
    }


def format_grading_prompt(question: str, answer: str, criterion: str) -> str:
    return GRADING_TEMPLATE.format(
        question=question,
        answer=answer,
        criterion=criterion,
        instructions=GRADING_INSTRUCTIONS,
    )


def parse_grade_from_response(response_text: str) -> tuple[bool, str]:
    match = re.search(r"GRADE:\s*([CI])", response_text, re.IGNORECASE)
    if match:
        grade = match.group(1).upper()
        return grade == "C", grade
    return False, "I"


def normalize_judge_specs(
    judge_specs: Iterable[ModelSpec] | None,
    *,
    default_specs: list[ModelSpec] | None = None,
) -> list[ModelSpec]:
    """Normalize judge specs, prefixing bare model ids with ``openrouter:``."""
    raw_specs = list(judge_specs) if judge_specs is not None else list(default_specs or [])
    normalized: list[ModelSpec] = []
    for spec in raw_specs:
        if isinstance(spec, str):
            stripped = spec.strip()
            if not stripped:
                continue
            normalized.append(stripped if ":" in stripped else f"openrouter:{stripped}")
        else:
            normalized.append(spec)
    return normalized


def resolve_judge_providers(judge_specs: Iterable[ModelSpec]) -> dict[str, Provider]:
    providers: dict[str, Provider] = {}
    for spec in judge_specs:
        provider = get_provider(spec)
        providers[f"{provider.provider_id}:{provider.model_name}"] = provider
    return providers


async def grade_with_provider(
    provider: Provider,
    question: str,
    answer: str,
    criterion: str,
    weight: float,
) -> dict[str, Any]:
    prompt = format_grading_prompt(question, answer, criterion)
    response = await provider.agenerate(prompt)
    passed, _ = parse_grade_from_response(response.text)
    return {
        "model": f"{provider.provider_id}:{provider.model_name}",
        "grade": CORRECT if passed else INCORRECT,
        "passed": passed,
        "awarded": weight if passed else 0.0,
        "explanation": response.text,
    }


async def grade_criterion_with_providers(
    criterion: dict[str, Any],
    *,
    question: str,
    submission: str,
    judge_providers: dict[str, Provider],
) -> dict[str, Any]:
    grader_type = criterion_grader_type(criterion)
    if grader_type == "ExactMatch":
        return grade_criterion_exact_match(criterion, submission)

    prompt = criterion_prompt(criterion)
    weight = float(criterion.get("weight", 0))
    if not judge_providers:
        raise ValueError("At least one judge provider is required for llm-judge criteria")

    judge_results = await asyncio.gather(
        *[
            grade_with_provider(provider, question, submission, prompt, weight)
            for provider in judge_providers.values()
        ]
    )
    passed_count = sum(1 for result in judge_results if result["passed"])
    majority_passed = passed_count > len(judge_results) / 2
    awarded = weight if majority_passed else 0.0
    return {
        "criterion_id": criterion.get("id"),
        "name": criterion.get("name"),
        "prompt": prompt,
        "semanticPrompt": prompt,
        "grader_type": "llm-judge",
        "weight": weight,
        "grade": CORRECT if majority_passed else INCORRECT,
        "passed": majority_passed,
        "awarded": awarded,
        "explanation": f"Majority vote: {passed_count}/{len(judge_results)} judges passed",
        "judges": list(judge_results),
    }


async def grade_criteria_with_providers(
    criteria: Iterable[dict[str, Any]],
    *,
    question: str,
    submission: str,
    judge_providers: dict[str, Provider],
) -> list[dict[str, Any]]:
    return await asyncio.gather(
        *[
            grade_criterion_with_providers(
                criterion,
                question=question,
                submission=submission,
                judge_providers=judge_providers,
            )
            for criterion in criteria
        ]
    )


def aggregate_scores(criteria_results: Iterable[dict[str, Any]]) -> float:
    return sum(float(result.get("awarded", 0.0) or 0.0) for result in criteria_results)


def normalize_score(total_score_raw: float) -> float:
    return total_score_raw / 100.0


async def evaluate_submission_with_providers(
    *,
    question: str,
    submission: str,
    criteria: Iterable[dict[str, Any]],
    pass_threshold: float,
    judge_providers: dict[str, Provider],
) -> dict[str, Any]:
    criteria_results = await grade_criteria_with_providers(
        list(criteria),
        question=question,
        submission=submission,
        judge_providers=judge_providers,
    )
    total_score_raw = aggregate_scores(criteria_results)
    return {
        "criteria_results": criteria_results,
        "total_score_raw": total_score_raw,
        "total_score": normalize_score(total_score_raw),
        "pass_threshold": pass_threshold,
        "passed": total_score_raw >= pass_threshold,
        "judge_names": list(judge_providers),
    }


def evaluate_submission_sync(
    *,
    question: str,
    submission: str,
    criteria: Iterable[dict[str, Any]],
    pass_threshold: float,
    judge_specs: Iterable[ModelSpec] | None = None,
    default_judge_specs: list[ModelSpec] | None = None,
) -> dict[str, Any]:
    normalized_specs = normalize_judge_specs(judge_specs, default_specs=default_judge_specs)
    judge_providers = resolve_judge_providers(normalized_specs) if normalized_specs else {}
    return asyncio.run(
        evaluate_submission_with_providers(
            question=question,
            submission=submission,
            criteria=list(criteria),
            pass_threshold=pass_threshold,
            judge_providers=judge_providers,
        )
    )

