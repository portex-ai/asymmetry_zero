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
    """
    Load an answer key file and return a mapping from task_id to its item.
    
    Reads a JSON file at `path` expecting a top-level list of items. Keeps only items that are dictionaries with a string `task_id` field. Ensures each kept item has a `criteria` field that is a list; if `criteria` is missing or not a non-empty list, the item will be returned with `criteria` set to an empty list.
    
    Parameters:
        path (str): Filesystem path to a JSON file containing a list of answer items.
    
    Returns:
        dict[str, dict[str, Any]]: A mapping from `task_id` to the corresponding item dictionary.
    """
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
    """
    Selects the best human-readable prompt for a grading criterion.
    
    Returns the first available text from the criterion in this precedence: `semanticPrompt`, `description`, `name`. If none are present or truthy, returns an empty string.
    
    Parameters:
        criterion (dict): A criterion object/dictionary that may contain `semanticPrompt`, `description`, or `name` keys.
    
    Returns:
        str: The chosen prompt text or an empty string if no prompt fields are available.
    """
    return (
        criterion.get("semanticPrompt")
        or criterion.get("description")
        or criterion.get("name")
        or ""
    )


def criterion_grader_type(criterion: dict[str, Any]) -> str:
    """
    Determine the grader type for a criterion, defaulting to "llm-judge".
    
    Parameters:
        criterion (dict): Criterion object that may contain a "grader_type" field.
    
    Returns:
        str: The grader type as a trimmed string; if the field is missing or falsy, returns "llm-judge".
    """
    return str(criterion.get("grader_type") or "llm-judge").strip()


def normalize_for_compare(text: str) -> str:
    """
    Normalize text for robust comparison by lowercasing, trimming, and collapsing internal whitespace.
    
    Parameters:
    	text (str): Input text to normalize.
    
    Returns:
    	normalized (str): The input converted to lowercase, leading/trailing whitespace removed, and consecutive internal whitespace collapsed to single spaces. Returns an empty string if `text` is not a non-empty string.
    """
    if not text or not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_final_answer(text: str) -> str:
    """
    Extracts the final answer from a submission text.
    
    If the text contains a final answer block labeled "Answer" (optionally prefixed with "Final"), returns the content of the last such block. Otherwise returns the last non-empty line. Returns an empty string for empty or whitespace-only input.
    
    Parameters:
        text (str): The submission text to extract the final answer from.
    
    Returns:
        str: The extracted final answer; empty string if none found.
    """
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
    """
    Evaluate a single grading criterion by checking whether the submission's final answer contains the criterion's reference prompt.
    
    Parameters:
        criterion (dict): Criterion metadata. Expected keys include:
            - "id": identifier for the criterion (optional).
            - "name": human-readable name (optional).
            - "weight": numeric weight (optional, defaults to 0).
            - fields used by criterion_prompt (e.g., "semanticPrompt", "description", "name").
        submission (str): Full submission text from which the final answer will be extracted.
    
    Returns:
        dict: Result object containing:
            - "criterion_id": criterion id or None.
            - "name": criterion name or None.
            - "prompt": the reference prompt used for matching (string).
            - "semanticPrompt": same as "prompt".
            - "grader_type": literal "ExactMatch".
            - "weight": weight as a float.
            - "grade": "C" if passed or "I" otherwise.
            - "passed": boolean indicating whether the reference was found in the extracted answer.
            - "awarded": numeric score awarded (weight if passed, otherwise 0.0).
            - "explanation": short text showing the reference and the extracted answer.
            - "judges": list with a single judge record (model "ExactMatch") mirroring grade/passed/awarded/explanation.
    """
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
    """
    Builds a full grading prompt by combining the question, the candidate answer, the criterion text, and the standard grading instructions.
    
    Parameters:
        question (str): The original question or task description.
        answer (str): The candidate's submitted answer.
        criterion (str): The human-readable criterion or rubric text to evaluate the answer against.
    
    Returns:
        str: A formatted prompt string ready to be sent to a grading/judge model.
    """
    return GRADING_TEMPLATE.format(
        question=question,
        answer=answer,
        criterion=criterion,
        instructions=GRADING_INSTRUCTIONS,
    )


def parse_grade_from_response(response_text: str) -> tuple[bool, str]:
    """
    Extract the "GRADE: C" or "GRADE: I" token from model output and interpret it as a correctness flag.
    
    Parameters:
        response_text (str): Text produced by a grader/judge to search for the grade token.
    
    Returns:
        tuple: (`True` if the found grade letter is `'C'`, otherwise `False`, grade_letter).
               If no grade token is present, returns `False` and `'I'`.
    """
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
    """
    Convert judge model specifications into a normalized list of ModelSpec entries.
    
    Strings in the input are trimmed; empty or whitespace-only strings are discarded. String specs that do not contain a colon are treated as bare model IDs and prefixed with "openrouter:". Non-string specs are preserved as-is. If `judge_specs` is None, `default_specs` is used instead (or an empty list when both are None).
    
    Parameters:
        judge_specs (Iterable[ModelSpec] | None): Iterable of model specifications (strings or spec objects) to normalize.
        default_specs (list[ModelSpec] | None): Fallback list of specs used when `judge_specs` is None.
    
    Returns:
        list[ModelSpec]: A list of normalized ModelSpec entries.
    """
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
    """
    Resolve model specifications into provider instances keyed by "provider_id:model_name".
    
    Parameters:
        judge_specs (Iterable[ModelSpec]): An iterable of model specifications (strings or spec objects) to resolve into providers.
    
    Returns:
        dict[str, Provider]: A mapping from the provider key "provider_id:model_name" to the corresponding Provider instance.
    """
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
    """
    Obtain a single-provider judgment for a criterion by sending a formatted grading prompt and parsing the provider's response.
    
    Returns:
        result (dict): Judgment summary with keys:
            - "model": provider identifier in the form "provider_id:model_name".
            - "grade": `"C"` for correct or `"I"` for incorrect.
            - "passed": `True` if the provider judged the criterion as satisfied, `False` otherwise.
            - "awarded": numeric score awarded for this criterion (full `weight` if passed, `0.0` otherwise).
            - "explanation": raw text of the provider's response.
    """
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
    """
    Grade a single criterion using either exact-match logic or one or more judge providers and return the criterion's grading result.
    
    Parameters:
        criterion (dict): Criterion definition; may include keys `id`, `name`, `weight` and grader-related fields.
        question (str): The original question or prompt associated with the submission.
        submission (str): The student's submitted answer text.
        judge_providers (dict[str, Provider]): Mapping of provider identifiers to Provider instances used for llm-judge criteria.
    
    Returns:
        dict: A result object containing:
            - criterion_id: criterion's `id` value (if present).
            - name: criterion's `name` value (if present).
            - prompt / semanticPrompt: the human-readable prompt used for grading.
            - grader_type: `"ExactMatch"` or `"llm-judge"`.
            - weight: numeric weight for the criterion.
            - grade: `CORRECT` ("C") if the criterion passed, `INCORRECT` ("I") otherwise.
            - passed: boolean indicating whether the criterion passed.
            - awarded: numeric score awarded (weight if passed, otherwise 0.0).
            - explanation: short human-readable explanation of the outcome.
            - judges: list of individual judge result records (one per provider) for llm-judge criteria.
    
    Raises:
        ValueError: If the criterion requires an LLM judge and `judge_providers` is empty.
    """
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
    """
    Grade multiple criteria concurrently using the given judge providers.
    
    Parameters:
        criteria (Iterable[dict[str, Any]]): An iterable of criterion dictionaries to evaluate.
        question (str): The original question or prompt associated with the submission.
        submission (str): The student's or agent's submitted answer text.
        judge_providers (dict[str, Provider]): Mapping of provider identifiers to Provider instances used to obtain judgments.
    
    Returns:
        list[dict[str, Any]]: A list of per-criterion grading results produced by grade_criterion_with_providers, preserving the order of `criteria`.
    """
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
    """
    Sum the awarded points from a sequence of criterion result dictionaries.
    
    Parameters:
        criteria_results (Iterable[dict[str, Any]]): An iterable of result dicts; each may include an "awarded" numeric value.
    
    Returns:
        float: The total of all `awarded` values (treats missing, None, or falsy values as 0.0).
    """
    return sum(float(result.get("awarded", 0.0) or 0.0) for result in criteria_results)


def normalize_score(total_score_raw: float) -> float:
    """
    Convert a raw total score on a 0–100 scale to a normalized score between 0 and 1.
    
    Parameters:
        total_score_raw (float): Total score expressed on a 0–100 scale.
    
    Returns:
        float: The normalized score (total_score_raw divided by 100.0).
    """
    return total_score_raw / 100.0


async def evaluate_submission_with_providers(
    *,
    question: str,
    submission: str,
    criteria: Iterable[dict[str, Any]],
    pass_threshold: float,
    judge_providers: dict[str, Provider],
) -> dict[str, Any]:
    """
    Evaluate a submission against a list of criteria using the provided judge providers and return per-criterion and aggregated results.
    
    Parameters:
        question (str): The original question or prompt associated with the submission.
        submission (str): The student's or agent's answer to be graded.
        criteria (Iterable[dict[str, Any]]): Iterable of criterion objects to evaluate.
        pass_threshold (float): Raw score threshold (same scale as `total_score_raw`) required to pass.
        judge_providers (dict[str, Provider]): Mapping of provider identifiers to judge provider instances used for model-based grading.
    
    Returns:
        result (dict[str, Any]): A dictionary containing:
            - criteria_results (list[dict[str, Any]]): Results for each criterion, including awarded scores and judge details.
            - total_score_raw (float): Sum of awarded points across criteria (raw scale).
            - total_score (float): Normalized total score (derived from `total_score_raw`).
            - pass_threshold (float): Echo of the provided pass threshold.
            - passed (bool): `true` if `total_score_raw` is greater than or equal to `pass_threshold`, `false` otherwise.
            - judge_names (list[str]): List of keys from `judge_providers` indicating which providers were consulted.
    """
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
    """
    Synchronously evaluate a submission against a set of grading criteria using resolved judge provider specifications.
    
    Parameters:
        question (str): The prompt or question associated with the submission.
        submission (str): The model output or student answer to be graded.
        criteria (Iterable[dict[str, Any]]): An iterable of criterion objects describing rubric items.
        pass_threshold (float): Raw score threshold (same scale as aggregate_scores output) required to mark the submission as passed.
        judge_specs (Iterable[ModelSpec] | None): Optional iterable of judge model specifications; strings without a provider prefix will be normalized (e.g., prefixed with "openrouter:") by normalize_judge_specs.
        default_judge_specs (list[ModelSpec] | None): Default judge specs used when `judge_specs` is None or empty.
    
    Returns:
        dict[str, Any]: Evaluation result containing:
            - criteria_results (list[dict]): Per-criterion grading results.
            - total_score_raw (float): Sum of awarded points (raw scale).
            - total_score (float): Normalized score (total_score_raw / 100.0).
            - pass_threshold (float): The provided pass threshold.
            - passed (bool): Whether total_score_raw meets or exceeds pass_threshold.
            - judge_names (list[str]): Keys identifying the resolved judge providers used for grading.
    """
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

