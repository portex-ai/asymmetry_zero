"""Writer for RL rewards JSON.

Implements extraction and writing of task scores to a JSON payload with
task_ids and reward arrays.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

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
