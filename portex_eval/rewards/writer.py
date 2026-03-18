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
    """
    Constructs a Rewards object from a sequence of (task_id, score) pairs.
    
    Parameters:
        task_scores (list[tuple[str, float]]): List of (task_id, score) tuples where `task_id` is a task identifier and `score` is its numeric reward.
    
    Returns:
        Rewards: A Rewards instance with `task_ids` and `reward` lists corresponding to the provided pairs.
    """
    task_ids = [task_id for task_id, _ in task_scores]
    rewards = [score for _, score in task_scores]
    return Rewards(task_ids=task_ids, reward=rewards)


def _task_reward_map(task_scores: list[tuple[str, float]]) -> dict[str, float]:
    """
    Create a mapping from task IDs to their numeric rewards.
    
    Parameters:
        task_scores (list[tuple[str, float]]): List of (task_id, score) pairs.
    
    Returns:
        dict[str, float]: Dictionary mapping each `task_id` to its `score`.
    """
    return {task_id: score for task_id, score in task_scores}


def _serialize_top_logprob(top_logprob: Any) -> dict[str, Any]:
    """
    Serialize a top-logprob candidate object into a plain dictionary.
    
    Parameters:
        top_logprob (Any): An object that may provide `token`, `logprob`, and `bytes` attributes.
    
    Returns:
        dict[str, Any]: Dictionary with keys `"token"`, `"logprob"`, and `"bytes"` whose values are taken from the corresponding attributes of `top_logprob` or `None` if an attribute is missing.
    """
    return {
        "token": getattr(top_logprob, "token", None),
        "logprob": getattr(top_logprob, "logprob", None),
        "bytes": getattr(top_logprob, "bytes", None),
    }


def _serialize_completion_logprobs(logprobs: Any) -> list[dict[str, Any]] | None:
    """
    Serialize a completion's logprob structure into a list of token-level dictionaries.
    
    Parameters:
        logprobs (Any): An object expected to have a `content` attribute that is a list of items; each item may have `token`, `logprob`, `bytes`, and `top_logprobs` attributes.
    
    Returns:
        list[dict[str, Any]] | None: A list where each dict contains `token`, `logprob`, `bytes`, and `top_logprobs` (a list of serialized candidate dicts) for each content item, or `None` if `logprobs` is `None` or its `content` is not a list.
    """
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
    """
    Serialize a content item into a JSON-serializable dictionary containing its type and any present fields.
    
    Parameters:
        item (Any): Object that may have attributes `type`, `text`, `image`, `document`, `reasoning`, and `detail`. Missing attributes are ignored.
    
    Returns:
        dict[str, Any]: Dictionary with a `type` key (uses the item's `type` value or `"unknown"` if absent) and entries for each of the present fields among `text`, `image`, `document`, `reasoning`, and `detail`.
    """
    item_type = getattr(item, "type", None)
    payload: dict[str, Any] = {"type": item_type or "unknown"}
    for field in ("text", "image", "document", "reasoning", "detail"):
        value = getattr(item, field, None)
        if value is not None:
            payload[field] = value
    return payload


def _serialize_message(message: Any) -> dict[str, Any]:
    """
    Serialize a message object into a dictionary containing its role and structured content.
    
    Parameters:
        message (Any): An object that may have `role` and `content` attributes. `content` can be a string (treated as a single text item), a list of content items, or other/absent.
    
    Returns:
        dict[str, Any]: A dictionary with:
            - "role": the message's role attribute or None.
            - "content": a list of serialized content items; for string content this is a single item `{"type": "text", "text": ...}`, for a list each item is serialized via _serialize_content_item, and for missing/unrecognized content an empty list is returned.
    """
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
    """
    Return a textual representation of a content value, joining texts and using placeholders for non-text items.
    
    If `content` is a string, it is returned unchanged. If `content` is a list, this returns the concatenation of each item's `text` (skipping empty strings) separated by newlines; items with `type` equal to `"image"` or `"document"` are represented as `"[image]"` or `"[document]"` respectively. For any other input type, an empty string is returned.
    
    Returns:
        text (str): Concatenated text and placeholders representing the provided content.
    """
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
    """
    Extracts non-assistant messages from a sample's messages list.
    
    Parameters:
        sample (Any): An object that may have a `messages` attribute (expected to be a list of message objects).
    
    Returns:
        list[Any]: A list containing messages whose `role` attribute is not "assistant". Returns an empty list if `messages` is missing or not a list.
    """
    messages = getattr(sample, "messages", None)
    if not isinstance(messages, list):
        return []
    return [message for message in messages if getattr(message, "role", None) != "assistant"]


def _prompt_text(messages: list[Any]) -> str:
    """
    Builds a single prompt string by concatenating non-empty messages as "role:\ncontent" blocks separated by blank lines.
    
    Parameters:
        messages (list[Any]): Sequence of message-like objects; each should expose a `role` attribute (string) and a `content` attribute (string or serializable content). Messages whose rendered content is empty are omitted.
    
    Returns:
        prompt_text (str): Concatenated prompt text where each message appears as:
        "role:
        content"
        separated by two newlines. Roles default to "unknown" when missing.
    """
    rendered: list[str] = []
    for message in messages:
        role = getattr(message, "role", None) or "unknown"
        content = _content_text(getattr(message, "content", None))
        if content.strip():
            rendered.append(f"{role}:\n{content}")
    return "\n\n".join(rendered)


def _sample_task_id(sample: Any) -> str:
    """
    Determine a task identifier for a sample by checking the sample's `id` then falling back to task IDs found in score metadata.
    
    If `sample.id` is a non-empty string, that value is returned. Otherwise, if `sample.scores` is a dict, the function searches each score's `metadata` for a non-empty string under the key `"task_id"` and returns the first match. Returns an empty string when no task identifier is found.
    
    Parameters:
        sample (Any): An object representing a sample; may have attributes `id` (str) or `scores` (dict of score-like objects whose `metadata` may contain `"task_id"`).
    
    Returns:
        str: The determined task identifier, or an empty string if none is available.
    """
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
    """
    Build a structured post-training payload by combining task rewards with an evaluation log.
    
    Parameters:
        task_scores (list[tuple[str, float]]): List of (task_id, score) tuples used to map rewards to samples.
        eval_log_path (str): Path to an evaluation log readable by inspect_ai.read_eval_log.
    
    Returns:
        dict: Payload with the following top-level keys:
            - "format": payload format identifier ("portex-rl-training-data").
            - "version": payload version number.
            - "source": dict containing "eval_log" (absolute path of the provided eval_log_path).
            - "records": list of record dicts, each containing:
                - "task_id": task identifier (str or empty string).
                - "sample_id": sample identifier or None.
                - "epoch": epoch number or None.
                - "model": model identifier or None.
                - "reward": numeric reward for the task_id or None.
                - "reference_file": reference file from sample metadata or None.
                - "prompt_messages": serialized list of prompt message dicts.
                - "prompt_text": concatenated prompt text (str).
                - "completion": completion text or object or None.
                - "prompt_token_ids": None (placeholder).
                - "completion_token_ids": None (placeholder).
                - "completion_logprobs": serialized completion logprobs list or None.
    
    Raises:
        ImportError: If the inspect_ai package (and its read_eval_log) is not available.
    """
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
    """
    Write task scores to the specified path as an RL rewards JSON file.
    
    Parameters:
        task_scores (list[tuple[str, float]]): List of (task_id, score) pairs to include in the payload.
        path (str): Destination file path for the JSON output.
    
    Returns:
        str: Absolute path to the written JSON file.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        payload = build_rewards(task_scores)
        json.dump({"task_ids": payload.task_ids, "reward": payload.reward}, handle, indent=2)

    return str(output_path.resolve())


def write_training_data(task_scores: list[tuple[str, float]], eval_log_path: str, path: str) -> str:
    """
    Write a structured training-data JSON file built from an evaluation log and task scores.
    
    Parameters:
        task_scores (list[tuple[str, float]]): List of (task_id, score) tuples used to annotate records in the training payload.
        eval_log_path (str): Path to the evaluation log used to build the training records.
        path (str): Filesystem path where the JSON payload will be written; parent directories will be created if needed.
    
    Returns:
        output_path (str): Absolute path to the written JSON file.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_training_data(task_scores, eval_log_path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return str(output_path.resolve())
