from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import pandas as pd
from inspect_ai.analysis import (
    EvalInfo,
    EvalModel,
    EventColumn,
    SampleScores,
    SampleSummary,
    evals_df,
    events_df,
    samples_df,
)


def _parse_json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return cast(dict[str, Any], parsed)
        return {}
    return {}


def _composite_id(dataset_location: str | None) -> str | None:
    if not dataset_location:
        return None
    return Path(dataset_location).parent.name


def _usage_for_models(model_usage: dict[str, Any], models: Iterable[str]) -> dict[str, int | None]:
    usage: dict[str, int | None] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost": None,
    }
    found = False
    for model_name in models:
        entry = model_usage.get(model_name, {})
        if not isinstance(entry, dict):
            continue
        found = True
        usage["input_tokens"] = (usage["input_tokens"] or 0) + int(
            entry.get("input_tokens", 0) or 0
        )
        usage["output_tokens"] = (usage["output_tokens"] or 0) + int(
            entry.get("output_tokens", 0) or 0
        )
        usage["total_tokens"] = (usage["total_tokens"] or 0) + int(
            entry.get("total_tokens", 0) or 0
        )
    return (
        usage
        if found
        else {"input_tokens": None, "output_tokens": None, "total_tokens": None, "cost": None}
    )


def _judge_models_from_metadata(metadata: dict[str, Any]) -> list[str]:
    judges: list[str] = []
    for criterion in metadata.get("criteria", []) or []:
        for judge in criterion.get("judges", []) or []:
            model = judge.get("model")
            if model and model not in judges:
                judges.append(model)
    return judges


def _event_call_cost(call: Any) -> float | None:
    if call is None:
        return None
    if isinstance(call, str):
        try:
            call = json.loads(call)
        except json.JSONDecodeError:
            return None
    if not isinstance(call, dict):
        return None
    for path in ("response", "request"):
        usage = call.get(path, {}).get("usage") if isinstance(call.get(path), dict) else None
        if isinstance(usage, dict) and "cost" in usage:
            try:
                cost_value = usage.get("cost")
                if cost_value is None:
                    return None
                return float(cost_value)
            except (TypeError, ValueError):
                return None
    return None


def _event_call_criterion(call: Any) -> str | None:
    if call is None:
        return None
    if isinstance(call, str):
        try:
            call = json.loads(call)
        except json.JSONDecodeError:
            return None
    if not isinstance(call, dict):
        return None
    request = call.get("request")
    if not isinstance(request, dict):
        return None
    messages = request.get("messages", [])
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        match = re.search(
            r"\n\*\*\*\n\[Criterion\]:\s*(.*?)\n\*\*\*\n\[END DATA\]",
            content,
            flags=re.DOTALL,
        )
        if match:
            return match.group(1).strip()
    return None


def _build_events_df(log_path: str) -> pd.DataFrame:
    columns = [
        EventColumn("model_event_model", path="model"),
        EventColumn("model_event_role", path="role"),
        EventColumn("model_event_time", path="output.time"),
        EventColumn("model_event_call", path="call"),
    ]
    events = events_df(logs=log_path, columns=columns)
    events = events[events["model_event_model"].notna()].copy()
    events["model_event_cost"] = events["model_event_call"].apply(_event_call_cost)
    events["model_event_criterion_prompt"] = events["model_event_call"].apply(
        _event_call_criterion
    )
    return events


def _filter_events(
    events: pd.DataFrame,
    role: str,
    fallback_models: Iterable[str] | None = None,
) -> pd.DataFrame:
    if events["model_event_role"].notna().any():
        return events[events["model_event_role"] == role]
    if fallback_models:
        return events[events["model_event_model"].isin(list(fallback_models))]
    return events.iloc[0:0]


def _summarize_events(events: pd.DataFrame) -> dict[str, Any]:
    if events.empty:
        return {"latency": None, "cost": None}
    latency = events["model_event_time"].dropna()
    cost = events["model_event_cost"].dropna()
    latency_value = float(latency.sum()) if not latency.empty else None
    cost_value = float(cost.sum()) if not cost.empty else None
    return {"latency": latency_value, "cost": cost_value}


def _matches_criterion(event_prompt: str | None, criterion_prompt: str | None) -> bool:
    if not isinstance(event_prompt, str) or not isinstance(criterion_prompt, str):
        return False
    return event_prompt.strip() == criterion_prompt.strip()


def _build_eval_level(
    log_row: pd.Series, tasks_df: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    dataset_location = log_row.get("dataset_location")
    composite_id = _composite_id(dataset_location)

    solver_model = log_row.get("model")
    eval_usage: dict[str, int | None] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost": None,
    }
    scorer_usage: dict[str, int | None] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost": None,
    }
    usage_found = False
    judge_models: set[str] = set()

    for _, row in tasks_df.iterrows():
        usage = _parse_json(row.get("model_usage"))
        if usage:
            usage_found = True
        if solver_model:
            solver_usage = _usage_for_models(usage, [solver_model])
            for key in eval_usage:
                if solver_usage.get(key) is not None:
                    eval_usage[key] = (eval_usage[key] or 0) + (solver_usage[key] or 0)

        metadata = _parse_json(row.get("score_portex_scorer_metadata"))
        for model in _judge_models_from_metadata(metadata):
            judge_models.add(model)
        judge_usage = _usage_for_models(usage, list(judge_models))
        for key in scorer_usage:
            if judge_usage.get(key) is not None:
                scorer_usage[key] = (scorer_usage[key] or 0) + (judge_usage[key] or 0)

    if not usage_found:
        eval_usage = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cost": None,
        }
        scorer_usage = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cost": None,
        }

    candidate_events = _filter_events(
        events, "candidate", [solver_model] if solver_model else None
    )
    grader_events = _filter_events(events, "grader", list(judge_models))
    candidate_summary = _summarize_events(candidate_events)
    grader_summary = _summarize_events(grader_events)

    return pd.DataFrame(
        [
            {
                "run_id": log_row.get("run_id"),
                "composite_id": composite_id,
                "log": log_row.get("log"),
                "created": log_row.get("created"),
                "packages": log_row.get("packages"),
                "task_file": log_row.get("task_file"),
                "model": log_row.get("model"),
                "model_base_url": log_row.get("model_base_url"),
                "model_arguments": log_row.get("model_args"),
                "model_generation_config": log_row.get("model_generate_config"),
                "dataset_location": dataset_location,
                "dataset_samples": log_row.get("dataset_samples"),
                "dataset_sample_ids": log_row.get("dataset_sample_ids"),
                "epochs": log_row.get("epochs"),
                "status": log_row.get("status"),
                "total_samples": log_row.get("total_samples"),
                "completed_samples": log_row.get("completed_samples"),
                "scored_headline_names": log_row.get("score_headline_name"),
                "scored_headline_metrics": log_row.get("score_headline_metric"),
                "score_headline_value": log_row.get("score_headline_value"),
                "score_headline_stderr": log_row.get("score_headline_stderr"),
                "solver_usage_data_latency": candidate_summary.get("latency"),
                "solver_usage_data_input_tokens": eval_usage.get("input_tokens") or 0,
                "solver_usage_data_output_tokens": eval_usage.get("output_tokens") or 0,
                "solver_usage_data_total_tokens": eval_usage.get("total_tokens") or 0,
                "solver_usage_data_cost": candidate_summary.get("cost"),
                "scorer_usage_data_latency": grader_summary.get("latency"),
                "scorer_usage_data_input_tokens": scorer_usage.get("input_tokens") or 0,
                "scorer_usage_data_output_tokens": scorer_usage.get("output_tokens") or 0,
                "scorer_usage_data_total_tokens": scorer_usage.get("total_tokens") or 0,
                "scorer_usage_data_cost": grader_summary.get("cost"),
            }
        ]
    )


def _build_task_level(
    log_row: pd.Series, tasks_df: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    dataset_location = log_row.get("dataset_location")
    composite_id = _composite_id(dataset_location)
    solver_model = log_row.get("model")

    rows: list[dict[str, Any]] = []
    for _, row in tasks_df.iterrows():
        metadata = _parse_json(row.get("score_portex_scorer_metadata"))
        usage = _parse_json(row.get("model_usage"))
        solver_usage = _usage_for_models(usage, [solver_model]) if solver_model else {}
        judge_models = _judge_models_from_metadata(metadata)
        judge_usage = _usage_for_models(usage, judge_models)
        sample_events = events[events["sample_id"] == row.get("sample_id")]
        candidate_events = _filter_events(
            sample_events, "candidate", [solver_model] if solver_model else None
        )
        grader_events = _filter_events(sample_events, "grader", judge_models)
        candidate_summary = _summarize_events(candidate_events)
        grader_summary = _summarize_events(grader_events)

        rows.append(
            {
                "run_id": row.get("run_id"),
                "composite_id": composite_id,
                "log": row.get("log"),
                "created": row.get("created"),
                "model": row.get("model"),
                "task_id": metadata.get("task_id"),
                "prompt": row.get("input"),
                "model_response": row.get("score_portex_scorer_answer"),
                "PassThreshold": metadata.get("pass_threshold"),
                "score": metadata.get("total_score"),
                "grade": row.get("score_portex_scorer"),
                "reasoning": row.get("score_portex_scorer_explanation"),
                "solver_usage_data_latency": candidate_summary.get("latency"),
                "solver_usage_data_input_tokens": solver_usage.get("input_tokens"),
                "solver_usage_data_output_tokens": solver_usage.get("output_tokens"),
                "solver_usage_data_total_tokens": solver_usage.get("total_tokens"),
                "solver_usage_data_cost": candidate_summary.get("cost"),
                "scorer_usage_data_latency": grader_summary.get("latency"),
                "scorer_usage_data_input_tokens": judge_usage.get("input_tokens"),
                "scorer_usage_data_output_tokens": judge_usage.get("output_tokens"),
                "scorer_usage_data_total_tokens": judge_usage.get("total_tokens"),
                "scorer_usage_data_cost": grader_summary.get("cost"),
            }
        )
    return pd.DataFrame(rows)


def _build_criterion_level(
    log_row: pd.Series, tasks_df: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    dataset_location = log_row.get("dataset_location")
    composite_id = _composite_id(dataset_location)
    rows: list[dict[str, Any]] = []

    for _, row in tasks_df.iterrows():
        metadata = _parse_json(row.get("score_portex_scorer_metadata"))
        usage = _parse_json(row.get("model_usage"))
        sample_events = events[events["sample_id"] == row.get("sample_id")]
        judge_models = _judge_models_from_metadata(metadata)
        judge_usage = _usage_for_models(usage, judge_models)
        for criterion in metadata.get("criteria", []) or []:
            prompt = criterion.get("prompt")
            criterion_events = sample_events[
                sample_events["model_event_criterion_prompt"].apply(
                    lambda event_prompt, p=prompt: _matches_criterion(event_prompt, p)
                )
            ]
            criterion_summary = _summarize_events(
                _filter_events(
                    criterion_events,
                    "grader",
                    _judge_models_from_metadata(metadata),
                )
            )
            rows.append(
                {
                    "run_id": row.get("run_id"),
                    "composite_id": composite_id,
                    "log": row.get("log"),
                    "created": row.get("created"),
                    "model": row.get("model"),
                    "task_id": metadata.get("task_id"),
                    "prompt": row.get("input"),
                    "model_response": row.get("score_portex_scorer_answer"),
                    "criterion_id": criterion.get("criterion_id"),
                    "criterion_name": criterion.get("name"),
                    "criterion_prompt": prompt,
                    "criteria_points": criterion.get("weight"),
                    "criteria_awarded": criterion.get("awarded"),
                    "criteria_passed": criterion.get("passed"),
                    "criteria_grade": criterion.get("grade"),
                    "scorer_usage_data_latency": criterion_summary.get("latency"),
                    "scorer_usage_data_input_tokens": judge_usage.get("input_tokens"),
                    "scorer_usage_data_output_tokens": judge_usage.get("output_tokens"),
                    "scorer_usage_data_total_tokens": judge_usage.get("total_tokens"),
                    "scorer_usage_data_cost": criterion_summary.get("cost"),
                }
            )
    return pd.DataFrame(rows)


def _build_judgement_level(
    log_row: pd.Series, tasks_df: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    dataset_location = log_row.get("dataset_location")
    composite_id = _composite_id(dataset_location)
    rows: list[dict[str, Any]] = []

    for _, row in tasks_df.iterrows():
        metadata = _parse_json(row.get("score_portex_scorer_metadata"))
        usage = _parse_json(row.get("model_usage"))
        sample_events = events[events["sample_id"] == row.get("sample_id")]
        for criterion in metadata.get("criteria", []) or []:
            for judge in criterion.get("judges", []) or []:
                judge_model = judge.get("model")
                judge_usage = _usage_for_models(usage, [judge_model]) if judge_model else {}
                prompt = criterion.get("prompt")
                judge_events = sample_events
                if judge_model:
                    judge_events = judge_events[judge_events["model_event_model"] == judge_model]
                judge_events = judge_events[
                    judge_events["model_event_criterion_prompt"].apply(
                        lambda event_prompt, p=prompt: _matches_criterion(event_prompt, p)
                    )
                ]
                judge_summary = _summarize_events(
                    _filter_events(
                        judge_events,
                        "grader",
                        [judge_model] if judge_model else None,
                    )
                )
                rows.append(
                    {
                        "run_id": row.get("run_id"),
                        "composite_id": composite_id,
                        "log": row.get("log"),
                        "created": row.get("created"),
                        "model": row.get("model"),
                        "task_id": metadata.get("task_id"),
                        "prompt": row.get("input"),
                        "model_response": row.get("score_portex_scorer_answer"),
                        "criterion_id": criterion.get("criterion_id"),
                        "criterion_name": criterion.get("name"),
                        "criterion_prompt": criterion.get("prompt"),
                        "criteria_points": criterion.get("weight"),
                        "judge_name": judge.get("model"),
                        "judge_awarded": judge.get("awarded"),
                        "judge_passed": judge.get("passed"),
                        "judge_grade": judge.get("grade"),
                        "judge_reasoning": judge.get("explanation"),
                        "scorer_usage_data_latency": judge_summary.get("latency"),
                        "scorer_usage_data_input_tokens": judge_usage.get("input_tokens"),
                        "scorer_usage_data_output_tokens": judge_usage.get("output_tokens"),
                        "scorer_usage_data_total_tokens": judge_usage.get("total_tokens"),
                        "scorer_usage_data_cost": judge_summary.get("cost"),
                    }
                )
    return pd.DataFrame(rows)


def run(log_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    logs_df = evals_df(log_path)
    tasks_df = samples_df(log_path, columns=EvalInfo + EvalModel + SampleSummary + SampleScores)
    events = _build_events_df(log_path)

    if logs_df.empty:
        raise ValueError(f"No eval log data found at {log_path}")

    log_row = logs_df.iloc[0]

    eval_level = _build_eval_level(log_row, tasks_df, events)
    task_level = _build_task_level(log_row, tasks_df, events)
    criterion_level = _build_criterion_level(log_row, tasks_df, events)
    judgement_level = _build_judgement_level(log_row, tasks_df, events)

    eval_level.to_csv(Path(output_dir) / "eval_level.csv", index=False)
    task_level.to_csv(Path(output_dir) / "task_level.csv", index=False)
    criterion_level.to_csv(Path(output_dir) / "criterion_level.csv", index=False)
    judgement_level.to_csv(Path(output_dir) / "judgement_level.csv", index=False)


def main() -> None:
    log_path = os.getenv("EVAL_LOG", "./runs/latest.eval")
    output_dir = os.getenv("OUTPUT_DIR", str(Path(log_path).parent))
    run(log_path, output_dir)


if __name__ == "__main__":
    main()
