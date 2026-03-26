"""Writer for RL rewards JSON.

Implements extraction and writing of task scores to a JSON payload with
task_ids and reward arrays.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from portex_eval.types import Rewards


def extract_rewards(task_level_csv: str) -> list[tuple[str, float]]:
    """Extract task_id and score pairs from a task_level.csv file.

    Args:
        task_level_csv: Path to the task_level.csv file.

    Returns:
        List of (task_id, score) tuples.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing or score is invalid.
    """
    csv_path = Path(task_level_csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"task_level.csv not found: {task_level_csv}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Empty or invalid CSV file: {task_level_csv}")
        if "task_id" not in reader.fieldnames:
            raise ValueError(f"task_level.csv missing 'task_id' column: {task_level_csv}")
        if "score" not in reader.fieldnames:
            raise ValueError(f"task_level.csv missing 'score' column: {task_level_csv}")

        rewards: list[tuple[str, float]] = []
        for row_idx, row in enumerate(reader):
            task_id = row.get("task_id")
            score_str = row.get("score")

            if task_id is None or task_id == "":
                continue
            if score_str is None or score_str == "":
                continue

            try:
                score = float(score_str)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid score value '{score_str}' at row {row_idx + 1}: {task_level_csv}"
                ) from exc

            rewards.append((task_id, score))

    return rewards


def build_rewards(task_scores: list[tuple[str, float]]) -> Rewards:
    """Build reward payload from task score tuples."""
    task_ids = [task_id for task_id, _ in task_scores]
    rewards = [score for _, score in task_scores]
    return Rewards(task_ids=task_ids, reward=rewards)


def _task_reward_map(task_scores: list[tuple[str, float]]) -> dict[str, float]:
    return {task_id: score for task_id, score in task_scores}


def _serialize_top_logprob(top_logprob: Any) -> dict[str, Any]:
    return {
        "token": getattr(top_logprob, "token", None),
        "logprob": getattr(top_logprob, "logprob", None),
        "bytes": getattr(top_logprob, "bytes", None),
    }


def _serialize_completion_logprobs(logprobs: Any) -> list[dict[str, Any]] | None:
    if logprobs is None:
        return None
    content = getattr(logprobs, "content", None)
    if not isinstance(content, list):
        return None
    serialized: list[dict[str, Any]] = []
    for item in content:
        top_logprobs = getattr(item, "top_logprobs", None)
        serialized.append(
            {
                "token": getattr(item, "token", None),
                "logprob": getattr(item, "logprob", None),
                "bytes": getattr(item, "bytes", None),
                "top_logprobs": (
                    [_serialize_top_logprob(candidate) for candidate in top_logprobs]
                    if isinstance(top_logprobs, list)
                    else None
                ),
            }
        )
    return serialized


def _serialize_content_item(item: Any) -> dict[str, Any]:
    item_type = getattr(item, "type", None)
    payload: dict[str, Any] = {"type": item_type or "unknown"}
    for field in ("text", "image", "document", "reasoning", "detail"):
        value = getattr(item, field, None)
        if value is not None:
            payload[field] = value
    return payload


def _serialize_message(message: Any) -> dict[str, Any]:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        serialized_content = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        serialized_content = [_serialize_content_item(item) for item in content]
    else:
        serialized_content = []
    return {
        "role": getattr(message, "role", None),
        "content": serialized_content,
    }


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text)
            continue
        item_type = getattr(item, "type", None)
        if item_type == "image":
            parts.append("[image]")
        elif item_type == "document":
            parts.append("[document]")
    return "\n".join(parts)


def _prompt_messages(sample: Any) -> list[Any]:
    messages = getattr(sample, "messages", None)
    if not isinstance(messages, list):
        return []
    return [message for message in messages if getattr(message, "role", None) != "assistant"]


def _prompt_text(messages: list[Any]) -> str:
    rendered: list[str] = []
    for message in messages:
        role = getattr(message, "role", None) or "unknown"
        content = _content_text(getattr(message, "content", None))
        if content.strip():
            rendered.append(f"{role}:\n{content}")
    return "\n\n".join(rendered)


def _sample_task_id(sample: Any) -> str:
    sample_id = getattr(sample, "id", None)
    if isinstance(sample_id, str) and sample_id:
        return sample_id

    scores = getattr(sample, "scores", None)
    if isinstance(scores, dict):
        for score in scores.values():
            metadata = getattr(score, "metadata", None)
            if isinstance(metadata, dict):
                task_id = metadata.get("task_id")
                if isinstance(task_id, str) and task_id:
                    return task_id
    return ""


def build_training_data(task_scores: list[tuple[str, float]], eval_log_path: str) -> dict[str, Any]:
    """Build structured post-training data from an eval log and rewards."""
    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:
        raise ImportError(
            "inspect-ai is required to build training artifacts from eval logs"
        ) from exc

    eval_log = read_eval_log(eval_log_path, header_only=False)
    reward_by_task = _task_reward_map(task_scores)
    records: list[dict[str, Any]] = []

    for sample in eval_log.samples:
        task_id = _sample_task_id(sample)
        prompt_messages = _prompt_messages(sample)
        output = getattr(sample, "output", None)
        choices = getattr(output, "choices", None) if output is not None else None
        first_choice = choices[0] if isinstance(choices, list) and choices else None

        records.append(
            {
                "task_id": task_id,
                "sample_id": getattr(sample, "id", None),
                "epoch": getattr(sample, "epoch", None),
                "model": getattr(output, "model", None),
                "reward": reward_by_task.get(task_id),
                "reference_file": (
                    getattr(sample, "metadata", {}).get("reference_file")
                    if isinstance(getattr(sample, "metadata", None), dict)
                    else None
                ),
                "prompt_messages": [_serialize_message(message) for message in prompt_messages],
                "prompt_text": _prompt_text(prompt_messages),
                "completion": getattr(output, "completion", None),
                "prompt_token_ids": None,
                "completion_token_ids": None,
                "completion_logprobs": (
                    _serialize_completion_logprobs(getattr(first_choice, "logprobs", None))
                    if first_choice is not None
                    else None
                ),
            }
        )

    return {
        "format": "portex-rl-training-data",
        "version": 1,
        "source": {
            "eval_log": str(Path(eval_log_path).resolve()),
        },
        "records": records,
    }


def write_rewards(task_scores: list[tuple[str, float]], path: str) -> str:
    """Write task scores to an rl_rewards.json file.

    Args:
        task_scores: List of (task_id, score) tuples.
        path: Output file path.

    Returns:
        The absolute path to the generated file.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        payload = build_rewards(task_scores)
        json.dump({"task_ids": payload.task_ids, "reward": payload.reward}, handle, indent=2)

    return str(output_path.resolve())


def write_training_data(task_scores: list[tuple[str, float]], eval_log_path: str, path: str) -> str:
    """Write structured post-training data to JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_training_data(task_scores, eval_log_path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return str(output_path.resolve())
