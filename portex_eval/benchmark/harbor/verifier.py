"""Harbor verifier entrypoint using the shared Portex grading core."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from portex_eval.grading import (
    DEFAULT_AGENT_JUDGE_MODELS,
    evaluate_submission_sync,
    normalize_judge_specs,
)

REWARD_PATH = Path("/logs/verifier/reward.json")
DETAIL_PATH = Path("/logs/verifier/portex_detail.json")
TESTS_DIR = Path("/tests")
CORRECT = "C"
INCORRECT = "I"


def load_task_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Task config not found: {config_path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Task config must be an object: {config_path}")
    config.setdefault("submission_path", "/app/answer.txt")
    config.setdefault("pass_threshold", 100)
    return config


def load_criteria(criteria_path: str) -> list[dict[str, Any]]:
    path = Path(criteria_path)
    if not path.exists():
        raise FileNotFoundError(f"Criteria config not found: {criteria_path}")
    criteria = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(criteria, list):
        raise ValueError(f"Criteria config must be a list: {criteria_path}")
    return [criterion for criterion in criteria if isinstance(criterion, dict)]


def resolve_judge_specs(task_config: dict[str, Any]) -> list[Any]:
    judge_configs = (os.environ.get("PORTEX_JUDGE_CONFIGS") or "").strip()
    if judge_configs:
        parsed = json.loads(judge_configs)
        if isinstance(parsed, list):
            return normalize_judge_specs(parsed, default_specs=DEFAULT_AGENT_JUDGE_MODELS)

    env_models = (os.environ.get("PORTEX_JUDGE_MODELS") or os.environ.get("JUDGE_MODELS") or "").strip()
    if env_models:
        if env_models.startswith("["):
            parsed = json.loads(env_models)
            if isinstance(parsed, list):
                return normalize_judge_specs(parsed, default_specs=DEFAULT_AGENT_JUDGE_MODELS)
        return normalize_judge_specs(env_models.split(","), default_specs=DEFAULT_AGENT_JUDGE_MODELS)

    task_models = task_config.get("judge_models")
    if isinstance(task_models, list):
        return normalize_judge_specs(task_models, default_specs=DEFAULT_AGENT_JUDGE_MODELS)

    return list(DEFAULT_AGENT_JUDGE_MODELS)


def build_detail_payload(
    *,
    task_config: dict[str, Any],
    submission: str,
    evaluation: dict[str, Any],
    error: str | None = None,
) -> dict[str, Any]:
    question = str(task_config.get("question") or "")
    pass_threshold = float(task_config.get("pass_threshold", 100))
    total_score_raw = float(evaluation.get("total_score_raw", 0.0) or 0.0)
    return {
        "task_id": task_config.get("task_id"),
        "question": question,
        "submission": submission,
        "reference_file": task_config.get("reference_file", ""),
        "pass_threshold": pass_threshold,
        "total_score": float(evaluation.get("total_score", 0.0) or 0.0),
        "total_score_raw": total_score_raw,
        "passed": bool(evaluation.get("passed", False)),
        "grade": CORRECT if evaluation.get("passed") else INCORRECT,
        "judge_names": evaluation.get("judge_names", []),
        "criteria_results": evaluation.get("criteria_results", []),
        "error": error,
    }


def write_detail(detail_path: str | Path, detail: dict[str, Any]) -> None:
    path = Path(detail_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(detail, indent=2), encoding="utf-8")


def write_reward(reward_path: str | Path, reward: float) -> None:
    path = Path(reward_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"reward": reward}, indent=2), encoding="utf-8")


def write_error_reward(
    reward_path: str | Path,
    detail_path: str | Path,
    *,
    task_config: dict[str, Any] | None,
    error_message: str,
) -> None:
    write_reward(reward_path, 0.0)
    write_detail(
        detail_path,
        build_detail_payload(
            task_config=task_config or {},
            submission="",
            evaluation={},
            error=error_message,
        ),
    )


def run_grading(
    task_config_path: str,
    criteria_path: str,
    reward_path: str,
    detail_path: str,
) -> None:
    task_config = load_task_config(task_config_path)
    criteria = load_criteria(criteria_path)
    submission_path = Path(str(task_config.get("submission_path", "/app/answer.txt")))
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission not found at {submission_path}")

    submission = submission_path.read_text(encoding="utf-8").strip()
    evaluation = evaluate_submission_sync(
        question=str(task_config["question"]),
        submission=submission,
        criteria=criteria,
        pass_threshold=float(task_config.get("pass_threshold", 100)),
        judge_specs=resolve_judge_specs(task_config),
        default_judge_specs=DEFAULT_AGENT_JUDGE_MODELS,
    )

    write_reward(reward_path, float(evaluation["total_score"]))
    write_detail(
        detail_path,
        build_detail_payload(
            task_config=task_config,
            submission=submission,
            evaluation=evaluation,
        ),
    )


def main() -> None:
    task_config_path = str(TESTS_DIR / "task_config.json")
    criteria_path = str(TESTS_DIR / "criteria.json")
    reward_path = str(REWARD_PATH)
    detail_path = str(DETAIL_PATH)

    try:
        run_grading(
            task_config_path=task_config_path,
            criteria_path=criteria_path,
            reward_path=reward_path,
            detail_path=detail_path,
        )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        task_config: dict[str, Any] | None = None
        if Path(task_config_path).exists():
            try:
                task_config = load_task_config(task_config_path)
            except Exception:  # noqa: BLE001
                task_config = None
        write_error_reward(
            reward_path,
            detail_path,
            task_config=task_config,
            error_message=str(exc),
        )
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
