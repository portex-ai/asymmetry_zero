"""Programmatic API for portex_eval."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from portex_eval.config import Config
from portex_eval.errors import PortexEvalError
from portex_eval.providers import get_provider, parse_model_string
from portex_eval.types import Benchmark, EvalResults, ReportPaths


def create_benchmark(path: str) -> Benchmark:
    """Create a Portex bundle from a BYOB JSON file.

    Args:
        path: Path to the benchmark.json input file.

    Returns:
        Benchmark with the generated bundle path and task count.
    """
    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        raise PortexEvalError(f"Benchmark JSON not found: {input_path}")

    payload = _load_json(input_path)
    if not isinstance(payload, list):
        raise PortexEvalError(
            f"Benchmark JSON must be a list of tasks, got {type(payload).__name__}: {input_path}"
        )

    output_dir = input_path.with_suffix("")
    if output_dir.exists() and output_dir.is_file():
        raise PortexEvalError(f"Output path is a file: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PortexEvalError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    refs_dir = output_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []

    for idx, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise PortexEvalError(
                f"Task entry at index {idx} must be an object: {input_path}"
            )
        task = entry.get("task")
        answer = entry.get("answer")
        reference_file = entry.get("reference_file") or ""

        if not isinstance(task, str) or not task.strip():
            raise PortexEvalError(f"task is required for entry {idx}: {input_path}")
        if not isinstance(answer, str):
            raise PortexEvalError(f"answer must be a string for entry {idx}: {input_path}")
        if not isinstance(reference_file, str):
            raise PortexEvalError(
                f"reference_file must be a string for entry {idx}: {input_path}"
            )

        task_id = str(uuid.uuid4())
        tasks.append(
            {
                "task_id": task_id,
                "task_prompt": task,
                "reference_file": reference_file,
            }
        )
        answers.append(
            {
                "task_id": task_id,
                "answer": answer,
                "reference_file": reference_file,
                "tools": [],
                "criteria": [],
                "passThreshold": 100,
            }
        )

        if reference_file:
            src_path = Path(reference_file)
            if not src_path.is_absolute():
                src_path = input_path.parent / src_path
            if not src_path.is_file():
                raise PortexEvalError(
                    f"reference_file not found for entry {idx}: {src_path}"
                )
            dest_path = refs_dir / reference_file
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest_path)

    _write_json(output_dir / "tasks.json", {"version": 2, "prompts": tasks})
    _write_json(output_dir / "answers.json", answers)

    return Benchmark(path=str(output_dir.resolve()), task_count=len(tasks))


def eval(
    *,
    path: str | None = None,
    benchmark: Benchmark | None = None,
    judges: list[str],
    candidates: list[str],
    output_dir: str | None = None,
    config: Config | None = None,
    task_spec: str | None = None,
) -> EvalResults:
    """Run an evaluation benchmark and return results."""
    from portex_eval.benchmark.run import benchmark_one

    if (path is None) == (benchmark is None):
        raise PortexEvalError("Provide exactly one of path or benchmark")
    if not judges:
        raise PortexEvalError("At least one judge model is required")
    if not candidates:
        raise PortexEvalError("At least one candidate model is required")

    if benchmark is None:
        if path is None:
            raise PortexEvalError("path is required when benchmark is not provided")
        bundle_path = Path(path).expanduser().resolve()
    else:
        bundle_path = benchmark.resolve_path()
    if not bundle_path.is_dir():
        raise PortexEvalError(f"Bundle directory not found: {bundle_path}")

    tasks_path = bundle_path / "tasks.json"
    answers_path = bundle_path / "answers.json"
    task_ids = _validate_tasks_json(tasks_path)
    _validate_answers_json(answers_path, task_ids)

    judge_models = [_validate_model_string(m, "judges") for m in judges]
    candidate_models = [_validate_model_string(m, "candidates") for m in candidates]

    cfg = config or Config.from_env()
    runs_root = Path(output_dir).expanduser().resolve() if output_dir else cfg.ensure_runs_dir()
    runs_root.mkdir(parents=True, exist_ok=True)

    previous_judge_models = os.environ.get("PORTEX_JUDGE_MODELS")
    os.environ["PORTEX_JUDGE_MODELS"] = ",".join(judge_models)

    eval_logs: list[str] = []
    last_run_id = ""
    last_output_dir = ""
    last_reports: ReportPaths | None = None
    last_rewards = ""

    try:
        for candidate in candidate_models:
            result = benchmark_one(
                bundle_dir=str(bundle_path),
                index_root=cfg.bundles_dir,
                eval_runs_root=str(runs_root),
                model_endpoint=candidate,
                task_spec=task_spec,
            )
            eval_logs.append(result.eval_log)
            last_run_id = result.run_id
            last_output_dir = result.output_dir

            reports_dir = Path(result.output_dir) / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            from portex_eval.reporting import tables as report_tables

            report_tables.run(result.eval_log, str(reports_dir))
            report_paths = ReportPaths(
                eval_level=str(reports_dir / "eval_level.csv"),
                task_level=str(reports_dir / "task_level.csv"),
                criterion_level=str(reports_dir / "criterion_level.csv"),
                judgement_level=str(reports_dir / "judgement_level.csv"),
            )
            rewards_path = _write_rewards(report_paths.task_level, Path(result.output_dir))

            last_reports = report_paths
            last_rewards = rewards_path
    finally:
        if previous_judge_models is None:
            os.environ.pop("PORTEX_JUDGE_MODELS", None)
        else:
            os.environ["PORTEX_JUDGE_MODELS"] = previous_judge_models

    if not last_run_id:
        raise PortexEvalError("No evaluation results produced")

    results = EvalResults(
        logs=eval_logs,
        reports=last_reports,
        rewards=last_rewards,
        run_id=last_run_id,
        output_dir=last_output_dir,
    )
    return results.with_absolute_paths()


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise PortexEvalError(
            f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _validate_model_string(model_string: str, field: str) -> str:
    try:
        provider_id, model_id = parse_model_string(model_string)
    except ValueError as exc:
        raise PortexEvalError(f"Invalid {field} model string '{model_string}': {exc}") from exc

    try:
        get_provider(model_string)
    except ValueError as exc:
        raise PortexEvalError(f"Unsupported {field} model '{model_string}': {exc}") from exc

    if provider_id == "openrouter":
        return f"openrouter/{model_id}"
    return f"{provider_id}/{model_id}"


def _validate_tasks_json(tasks_path: Path) -> set[str]:
    if not tasks_path.is_file():
        raise PortexEvalError(f"tasks.json not found: {tasks_path}")

    payload = _load_json(tasks_path)
    if isinstance(payload, dict):
        version = payload.get("version")
        if version != 2:
            raise PortexEvalError(
                f"tasks.json must have version=2 (got {version!r}): {tasks_path}"
            )
        prompts = payload.get("prompts")
        if not isinstance(prompts, list):
            raise PortexEvalError(f"tasks.json prompts must be a list: {tasks_path}")
        records = prompts
    elif isinstance(payload, list):
        records = payload
    else:
        raise PortexEvalError(
            f"tasks.json must be a list or object, got {type(payload).__name__}: {tasks_path}"
        )

    task_ids: set[str] = set()
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise PortexEvalError(f"tasks.json entry {idx} must be an object: {tasks_path}")
        task_id = record.get("task_id")
        task_prompt = record.get("task_prompt") or record.get("prompt") or record.get("task")
        if not isinstance(task_id, str) or not task_id.strip():
            raise PortexEvalError(f"tasks.json entry {idx} missing task_id: {tasks_path}")
        if not isinstance(task_prompt, str) or not task_prompt.strip():
            raise PortexEvalError(
                f"tasks.json entry {idx} missing task_prompt/prompt: {tasks_path}"
            )
        task_ids.add(task_id)
    return task_ids


def _validate_answers_json(answers_path: Path, task_ids: set[str]) -> None:
    if not answers_path.is_file():
        raise PortexEvalError(f"answers.json not found: {answers_path}")

    payload = _load_json(answers_path)
    if not isinstance(payload, list):
        raise PortexEvalError(
            f"answers.json must be a list, got {type(payload).__name__}: {answers_path}"
        )

    for idx, record in enumerate(payload):
        if not isinstance(record, dict):
            raise PortexEvalError(
                f"answers.json entry {idx} must be an object: {answers_path}"
            )
        task_id = record.get("task_id")
        answer = record.get("answer")
        if not isinstance(task_id, str) or not task_id.strip():
            raise PortexEvalError(f"answers.json entry {idx} missing task_id: {answers_path}")
        if task_id not in task_ids:
            raise PortexEvalError(
                f"answers.json entry {idx} references unknown task_id '{task_id}': {answers_path}"
            )
        if not isinstance(answer, str):
            raise PortexEvalError(
                f"answers.json entry {idx} answer must be a string: {answers_path}"
            )


def _write_rewards(task_level_csv: str, output_dir: Path) -> str:
    import csv

    rewards_path = output_dir / "rl_rewards.txt"
    with open(task_level_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if (
            not reader.fieldnames
            or "task_id" not in reader.fieldnames
            or "score" not in reader.fieldnames
        ):
            raise PortexEvalError(
                f"task_level.csv missing task_id/score columns: {task_level_csv}"
            )
        rows = list(reader)

    with rewards_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            task_id = row.get("task_id")
            score = row.get("score")
            if task_id is None or score is None:
                continue
            handle.write(f"{task_id} {score}\n")
    return str(rewards_path)
