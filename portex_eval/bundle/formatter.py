"""Bundle formatter for converting bundles to v2 format with criteria induction.

Converts input bundles (tasks.json + answers.json + refs/) to the standardized
v2 format, optionally using an LLM to induce evaluation criteria from answers.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from portex_eval.errors import PortexEvalError
from portex_eval.providers import Provider, get_provider

DEFAULT_CRITERIA_INDUCTION_MODEL = "openrouter:deepseek/deepseek-v3.1-terminus"
ALLOWED_GRADER_TYPES = {"ExactMatch", "llm-judge"}

CRITERIA_INDUCTION_PROMPT = """\
You are an expert at creating evaluation criteria for AI assistant responses.

Given the following task and reference answer, generate evaluation criteria that
can be used to assess whether a candidate response is correct and complete.

[TASK]
{task_prompt}

[REFERENCE ANSWER]
{answer}

Generate 2-5 evaluation criteria. Each criterion should:
1. Be specific and measurable
2. Focus on one aspect of correctness or quality
3. Have a weight reflecting its importance (weights must sum to exactly 100)

Respond with a JSON array of criteria objects. Each object must have:
- "id": A unique short identifier (lowercase, no spaces, use underscores)
- "name": A brief human-readable name
- "weight": An integer weight (1-100)
- "semanticPrompt": A detailed description of what to check for

Example format:
```json
[
  {{"id": "factual_accuracy", "name": "Factual Accuracy", "weight": 40,
    "semanticPrompt": "The response contains the correct factual info..."}},
  {{"id": "completeness", "name": "Completeness", "weight": 35,
    "semanticPrompt": "The response addresses all parts of the question..."}},
  {{"id": "clarity", "name": "Clarity", "weight": 25,
    "semanticPrompt": "The response is clear and easy to understand..."}}
]
```

Respond ONLY with the JSON array, no additional text or markdown formatting.
"""


@dataclass
class FormattedBundle:
    """Result of formatting a bundle.

    Attributes:
        output_dir: Absolute path to the formatted bundle directory.
        task_count: Number of tasks in the bundle.
        criteria_induced: Whether criteria were induced via LLM.
    """

    output_dir: str
    task_count: int
    criteria_induced: bool


def format_bundle(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    induce_criteria_flag: bool = False,
    judge_model: str = DEFAULT_CRITERIA_INDUCTION_MODEL,
) -> FormattedBundle:
    """Format a bundle to v2 format with optional criteria induction.

    Reads an input bundle (tasks.json + answers.json + refs/) and produces
    a formatted bundle in the output directory. Validates input schema and
    generates tasks.json v2 format.

    Args:
        input_path: Path to the input bundle directory. Must contain tasks.json
            and answers.json. May contain refs/ directory.
        output_dir: Path to the output bundle directory. Will be created if
            it doesn't exist. Must be empty if it exists.
        induce_criteria_flag: If True, use an LLM to generate evaluation criteria
            from the reference answers. Criteria weights will sum to 100.
        judge_model: Model string for criteria induction
            (e.g., 'openrouter:google/gemini-2.5-flash').
            Only used when induce_criteria_flag is True.

    Returns:
        FormattedBundle with output path and metadata.

    Raises:
        PortexEvalError: If input validation fails or output directory is not empty.
    """
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    _validate_input_bundle(input_path)
    _prepare_output_dir(output_dir)

    tasks_data = _load_json(input_path / "tasks.json")
    answers_data = _load_json(input_path / "answers.json")

    tasks = _normalize_tasks(tasks_data, input_path / "tasks.json")
    answers = _normalize_answers(answers_data, input_path / "answers.json")

    task_ids = {task["task_id"] for task in tasks}
    _validate_answers_task_ids(answers, task_ids, input_path / "answers.json")

    if induce_criteria_flag:
        provider = get_provider(judge_model)
        task_prompts = {task["task_id"]: task["task_prompt"] for task in tasks}
        answers = _induce_criteria_for_answers(answers, task_prompts, provider)
    answers = _ensure_answers_have_criteria(answers, input_path / "answers.json")

    _write_tasks_v2(output_dir / "tasks.json", tasks)
    _write_answers(output_dir / "answers.json", answers)
    _copy_refs(input_path, output_dir)

    return FormattedBundle(
        output_dir=str(output_dir),
        task_count=len(tasks),
        criteria_induced=induce_criteria_flag,
    )


def induce_criteria(
    answer: str,
    task_prompt: str,
    judge_model: str = DEFAULT_CRITERIA_INDUCTION_MODEL,
) -> list[dict[str, Any]]:
    """Induce evaluation criteria from a reference answer using an LLM.

    Uses the specified judge model to generate criteria based on the task
    and reference answer. The criteria weights will sum to exactly 100.

    Args:
        answer: The reference answer to analyze.
        task_prompt: The original task prompt/question.
        judge_model: Model string for criteria induction.

    Returns:
        List of criteria dictionaries, each with id, name, weight, and semanticPrompt.

    Raises:
        PortexEvalError: If criteria generation or parsing fails.
    """
    provider = get_provider(judge_model)
    return _induce_criteria_single(answer, task_prompt, provider)


def _validate_input_bundle(input_path: Path) -> None:
    """Validate that input path contains required files."""
    if not input_path.is_dir():
        raise PortexEvalError(f"Input bundle directory not found: {input_path}")

    tasks_path = input_path / "tasks.json"
    if not tasks_path.is_file():
        raise PortexEvalError(f"tasks.json not found in input bundle: {tasks_path}")

    answers_path = input_path / "answers.json"
    if not answers_path.is_file():
        raise PortexEvalError(f"answers.json not found in input bundle: {answers_path}")


def _prepare_output_dir(output_dir: Path) -> None:
    """Prepare output directory, ensuring it's empty or creatable."""
    if output_dir.exists():
        if output_dir.is_file():
            raise PortexEvalError(f"Output path exists as a file: {output_dir}")
        if any(output_dir.iterdir()):
            raise PortexEvalError(f"Output directory is not empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    refs_dir = output_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Any:
    """Load JSON from file with error handling."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise PortexEvalError(
            f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc


def _write_json(path: Path, data: Any) -> None:
    """Write JSON to file with consistent formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _normalize_tasks(data: Any, source_path: Path) -> list[dict[str, Any]]:
    """Normalize tasks data to a list of task records.

    Handles both  (list) and v (object with version and prompts) formats.
    """
    if isinstance(data, dict):
        prompts = data.get("prompts")
        version = data.get("version")
        if version is not None and not isinstance(version, int):
            raise PortexEvalError(
                f"tasks.json version must be an integer when provided: {source_path}"
            )
        if not isinstance(prompts, list):
            raise PortexEvalError(f"tasks.json prompts must be a list: {source_path}")
        records = prompts
    elif isinstance(data, list):
        records = data
    else:
        raise PortexEvalError(
            f"tasks.json must be a list or object, got {type(data).__name__}: {source_path}"
        )

    tasks: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise PortexEvalError(f"tasks.json entry {idx} must be an object: {source_path}")

        task_id = record.get("task_id")
        task_prompt = record.get("task_prompt") or record.get("prompt") or record.get("task") or ""
        reference_file = record.get("reference_file", "")

        if not isinstance(task_id, str) or not task_id.strip():
            raise PortexEvalError(f"tasks.json entry {idx} missing task_id: {source_path}")
        if not isinstance(task_prompt, str) or not task_prompt.strip():
            raise PortexEvalError(
                f"tasks.json entry {idx} missing task_prompt/prompt/task: {source_path}"
            )

        tasks.append(
            {
                "task_id": task_id,
                "task_prompt": task_prompt,
                "reference_file": reference_file if isinstance(reference_file, str) else "",
            }
        )

    return tasks


def _normalize_answers(data: Any, source_path: Path) -> list[dict[str, Any]]:
    """
    Normalize raw answers.json content into a list of validated answer records.
    
    Parameters:
        data (Any): Parsed JSON content of answers.json; expected to be a list of answer objects.
        source_path (Path): Path to the source answers.json file, used in error messages.
    
    Returns:
        list[dict[str, Any]]: A list of normalized answer records. Each record contains:
            - "task_id" (str): validated non-empty task identifier.
            - "answer" (str|None): the original answer text or None if not provided.
            - "reference_file" (str): reference filename or empty string.
            - "tools" (list): list of tools (may be empty).
            - "criteria" (list[dict]): validated and normalized criteria entries.
            - "passThreshold" (int|float): pass threshold (defaults to 100).
    
    Raises:
        PortexEvalError: If the top-level data is not a list, an entry is not an object,
                         a task_id is missing or empty, an answer is present but not a string,
                         criteria is not a list, or any criterion fails validation.
    """
    if not isinstance(data, list):
        raise PortexEvalError(
            f"answers.json must be a list, got {type(data).__name__}: {source_path}"
        )

    answers: list[dict[str, Any]] = []
    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            raise PortexEvalError(f"answers.json entry {idx} must be an object: {source_path}")

        task_id = record.get("task_id")
        answer = record.get("answer")

        if not isinstance(task_id, str) or not task_id.strip():
            raise PortexEvalError(f"answers.json entry {idx} missing task_id: {source_path}")
        if answer is not None and not isinstance(answer, str):
            raise PortexEvalError(
                f"answers.json entry {idx} answer must be a string when provided: {source_path}"
            )

        criteria = record.get("criteria", [])
        if criteria is None:
            criteria = []
        if not isinstance(criteria, list):
            raise PortexEvalError(
                f"answers.json entry {idx} criteria must be a list: {source_path}"
            )
        normalized_criteria = [
            _normalize_criterion(
                criterion,
                context=f"answers.json entry {idx} criterion {criterion_idx}",
                source_path=source_path,
            )
            for criterion_idx, criterion in enumerate(criteria)
        ]

        answers.append(
            {
                "task_id": task_id,
                "answer": answer,
                "reference_file": record.get("reference_file", ""),
                "tools": record.get("tools", []),
                "criteria": normalized_criteria,
                "passThreshold": record.get("passThreshold", 100),
            }
        )

    return answers


def _validate_answers_task_ids(
    answers: list[dict[str, Any]],
    task_ids: set[str],
    source_path: Path,
) -> None:
    """Validate that all answer task_ids reference existing tasks."""
    for idx, answer in enumerate(answers):
        task_id = answer["task_id"]
        if task_id not in task_ids:
            raise PortexEvalError(
                f"answers.json entry {idx} references unknown task_id '{task_id}': {source_path}"
            )


def _write_tasks_v2(path: Path, tasks: list[dict[str, Any]]) -> None:
    """Write tasks in v2 format."""
    _write_json(path, {"version": 2, "prompts": tasks})


def _write_answers(path: Path, answers: list[dict[str, Any]]) -> None:
    """
    Write answers to a JSON file omitting raw answer text.
    
    Each entry in `answers` is serialized with the "answer" key removed; the resulting list is written to `path` as JSON.
    """
    serialized_answers = []
    for answer in answers:
        serialized_answers.append({k: v for k, v in answer.items() if k != "answer"})
    _write_json(path, serialized_answers)


def _copy_refs(input_path: Path, output_dir: Path) -> None:
    """
    Copy visible files and directories from the input bundle's refs directory into the output bundle's refs directory.
    
    If the input refs directory does not exist, the function does nothing. Items whose names start with "." are skipped. Directory contents are merged into existing destination directories; file metadata is preserved when copied.
    
    Parameters:
        input_path (Path): Path to the input bundle root containing a `refs/` subdirectory.
        output_dir (Path): Path to the output bundle root where `refs/` contents will be copied.
    """
    input_refs = input_path / "refs"
    output_refs = output_dir / "refs"

    if not input_refs.is_dir():
        return

    for item in input_refs.iterdir():
        if item.name.startswith("."):
            continue

        dest = output_refs / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)


def _induce_criteria_for_answers(
    answers: list[dict[str, Any]],
    task_prompts: dict[str, str],
    provider: Provider,
) -> list[dict[str, Any]]:
    """
    Induces evaluation criteria for answers that lack them using the provided LLM provider.
    
    For each answer in `answers` that has an empty or missing `criteria` list, generates criteria by calling the given `provider` with the corresponding task prompt and answer text, and returns a new list where those answers have a populated `criteria` field. Answers that already include criteria are returned unchanged.
    
    Parameters:
        answers (list[dict]): List of answer records; each record must include `task_id` and may include `answer` and `criteria`.
        task_prompts (dict[str, str]): Mapping from task_id to the task prompt text used when inducing criteria.
        provider (Provider): LLM provider used to generate criteria.
    
    Returns:
        list[dict]: New list of answer records where missing criteria have been added. Each added `criteria` value is a non-empty list of criterion dictionaries.
    
    Raises:
        PortexEvalError: If an answer requires induced criteria but its `answer` field is missing, not a string, or blank.
    """
    updated_answers: list[dict[str, Any]] = []

    for answer in answers:
        existing_criteria = answer.get("criteria", [])
        if existing_criteria:
            updated_answers.append(answer)
            continue

        task_id = answer["task_id"]
        task_prompt = task_prompts.get(task_id, "")
        answer_text = answer["answer"]
        if not isinstance(answer_text, str) or not answer_text.strip():
            raise PortexEvalError(
                f"answers.json entry for task_id '{task_id}' requires answer text to induce criteria"
            )

        criteria = _induce_criteria_single(answer_text, task_prompt, provider)

        updated_answer = answer.copy()
        updated_answer["criteria"] = criteria
        updated_answers.append(updated_answer)

    return updated_answers


def _ensure_answers_have_criteria(
    answers: list[dict[str, Any]],
    source_path: Path,
) -> list[dict[str, Any]]:
    """
    Verify that every answer entry contains a non-empty list under the "criteria" key.
    
    Parameters:
        answers (list[dict[str, Any]]): List of answer records to validate. Each record is expected to include a "criteria" key with a non-empty list.
        source_path (Path): Path to the source answers file; included in error messages for traceability.
    
    Returns:
        list[dict[str, Any]]: The same list passed in, returned unchanged when validation succeeds.
    
    Raises:
        PortexEvalError: If any answer's "criteria" is missing, not a list, or an empty list. The error message includes the offending entry index and the source_path.
    """
    for idx, answer in enumerate(answers):
        criteria = answer.get("criteria", [])
        if not isinstance(criteria, list) or not criteria:
            raise PortexEvalError(
                f"answers.json entry {idx} criteria must be a non-empty list: {source_path}"
            )
    return answers


def _induce_criteria_single(
    answer: str,
    task_prompt: str,
    provider: Provider,
) -> list[dict[str, Any]]:
    """
    Induces evaluation criteria for a single answer using the configured LLM provider.
    
    Formats a prompt from the task prompt and answer, sends it to the provider, parses the LLM response
    into a list of criterion objects, ensures each criterion has a `grader_type` (defaulting to
    "llm-judge" when absent), and normalizes criterion weights to sum to 100.
    
    Parameters:
        answer (str): The answer text for which criteria should be induced.
        task_prompt (str): The original task prompt to provide context to the LLM.
    
    Returns:
        list[dict]: A list of criterion dictionaries. Each criterion includes keys such as `id`,
        `name`, `semanticPrompt`/`description` (at least one), `weight` (numeric; weights sum to 100),
        and `grader_type`.
    
    Raises:
        PortexEvalError: If LLM generation or response parsing fails.
    """
    prompt = CRITERIA_INDUCTION_PROMPT.format(
        task_prompt=task_prompt,
        answer=answer,
    )

    response = provider.generate(prompt, temperature=0.3, max_tokens=2000)
    criteria = _parse_criteria_response(response.text)
    criteria = [{**criterion, "grader_type": criterion.get("grader_type", "llm-judge")} for criterion in criteria]
    criteria = _normalize_criteria_weights(criteria)

    return criteria


def _parse_criteria_response(response_text: str) -> list[dict[str, Any]]:
    """
    Extract a JSON array of criteria from an LLM response text.
    
    Parses the response (including JSON inside fenced code blocks) and returns the decoded list of criterion objects.
    
    Returns:
        criteria (list[dict]): A list of criterion objects parsed from the response.
    
    Raises:
        PortexEvalError: If the JSON cannot be parsed, if the top-level value is not a list,
                         if any item is not an object, or if any criterion is missing a required
                         field ('id', 'name', 'weight', 'semanticPrompt').
    """
    text = response_text.strip()

    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if json_match:
        text = json_match.group(1).strip()

    if text.startswith("["):
        pass
    else:
        bracket_start = text.find("[")
        bracket_end = text.rfind("]")
        if bracket_start != -1 and bracket_end > bracket_start:
            text = text[bracket_start : bracket_end + 1]

    try:
        criteria = json.loads(text)
    except json.JSONDecodeError as exc:
        preview = response_text[:500]
        raise PortexEvalError(
            f"Failed to parse criteria from LLM response: {exc.msg}. Response: {preview}"
        ) from exc

    if not isinstance(criteria, list):
        preview = response_text[:500]
        raise PortexEvalError(
            f"Criteria must be a list, got {type(criteria).__name__}. Response: {preview}"
        )

    for idx, criterion in enumerate(criteria):
        if not isinstance(criterion, dict):
            raise PortexEvalError(f"Criterion {idx} must be an object: {criterion}")

        required_fields = ["id", "name", "weight", "semanticPrompt"]
        for field in required_fields:
            if field not in criterion:
                raise PortexEvalError(f"Criterion {idx} missing required field '{field}'")

    return criteria


def _normalize_criterion(
    criterion: Any,
    *,
    context: str,
    source_path: Path,
) -> dict[str, Any]:
    """
    Validate and normalize a single criterion object.
    
    Parameters:
        criterion (Any): The raw criterion value to validate and normalize.
        context (str): Human-readable context used in error messages (e.g., field path).
        source_path (Path): Source file path included in error messages for traceability.
    
    Returns:
        dict[str, Any]: A normalized criterion dictionary with a validated `grader_type` (defaults to "llm-judge" when absent).
    
    Raises:
        PortexEvalError: If `criterion` is not an object; if `id` is missing or empty; if `weight` is not numeric; if `grader_type` is not one of the allowed values; or if none of `semanticPrompt`, `description`, or `name` is a non-empty string.
    """
    if not isinstance(criterion, dict):
        raise PortexEvalError(f"{context} must be an object: {source_path}")

    criterion_id = criterion.get("id")
    if not isinstance(criterion_id, str) or not criterion_id.strip():
        raise PortexEvalError(f"{context} missing id: {source_path}")

    weight = criterion.get("weight")
    if not isinstance(weight, int | float):
        raise PortexEvalError(f"{context} weight must be numeric: {source_path}")

    grader_type = criterion.get("grader_type", "llm-judge")
    if grader_type not in ALLOWED_GRADER_TYPES:
        allowed = ", ".join(sorted(ALLOWED_GRADER_TYPES))
        raise PortexEvalError(f"{context} grader_type must be one of {allowed}: {source_path}")

    prompt_fields = [
        criterion.get("semanticPrompt"),
        criterion.get("description"),
        criterion.get("name"),
    ]
    if not any(isinstance(field, str) and field.strip() for field in prompt_fields):
        raise PortexEvalError(
            f"{context} requires one of semanticPrompt, description, or name: {source_path}"
        )

    return {**criterion, "grader_type": grader_type}


def _normalize_criteria_weights(criteria: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize a list of criterion dictionaries so their weights sum to exactly 100.
    
    Parameters:
        criteria (list[dict[str, Any]]): List of criterion objects; each may include a numeric `weight` field.
    
    Returns:
        list[dict[str, Any]]: A new list of criterion objects with numeric `weight` values adjusted so their sum is exactly 100.
            - If all input weights sum to 0, weights are distributed evenly and any remainder is given to earlier items.
            - Otherwise, weights are scaled proportionally and rounded; the final item's weight is adjusted to ensure the total equals 100.
            - All other fields on each criterion are preserved.
    """
    if not criteria:
        return criteria

    total_weight = sum(float(c.get("weight", 0)) for c in criteria)

    if total_weight == 0:
        equal_weight = 100 // len(criteria)
        remainder = 100 % len(criteria)
        normalized = []
        for idx, criterion in enumerate(criteria):
            weight = equal_weight + (1 if idx < remainder else 0)
            normalized.append({**criterion, "weight": weight})
        return normalized

    normalized = []
    running_total = 0
    for idx, criterion in enumerate(criteria):
        raw_weight = float(criterion.get("weight", 0))
        if idx == len(criteria) - 1:
            weight = 100 - running_total
        else:
            weight = round((raw_weight / total_weight) * 100)
            running_total += weight
        normalized.append({**criterion, "weight": weight})

    return normalized
