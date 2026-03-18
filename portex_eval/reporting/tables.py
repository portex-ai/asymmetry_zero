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
    """
    Extracts unique judge model names from evaluation metadata.
    
    Parameters:
        metadata (dict[str, Any]): Evaluation metadata containing a "criteria" sequence, where each criterion may include a "judges" sequence of objects with a "model" key.
    
    Returns:
        list[str]: Unique judge model names in the order they first appear; an empty list if no judge models are present.
    """
    judges: list[str] = []
    for criterion in metadata.get("criteria", []) or []:
        for judge in criterion.get("judges", []) or []:
            model = judge.get("model")
            if model and model not in judges:
                judges.append(model)
    return judges


def _score_prefix(tasks_df: pd.DataFrame) -> str:
    """
    Determine the score field prefix to use from task dataframe column names.
    
    Scans tasks_df for columns matching the pattern "score_<prefix>_metadata" and returns a chosen prefix. Prefers "score_portex_scorer", then "score_provider_scorer", otherwise returns the first discovered prefix; if none found, returns "score_portex_scorer".
    
    Parameters:
        tasks_df (pd.DataFrame): DataFrame of task rows that may contain score-related columns named like "score_<prefix>_metadata".
    
    Returns:
        str: The selected score prefix (including the "score_" prefix), e.g. "score_portex_scorer".
    """
    prefixes = sorted(
        {
            column[: -len("_metadata")]
            for column in tasks_df.columns
            if column.startswith("score_") and column.endswith("_metadata")
        }
    )
    if "score_portex_scorer" in prefixes:
        return "score_portex_scorer"
    if "score_provider_scorer" in prefixes:
        return "score_provider_scorer"
    if prefixes:
        return prefixes[0]
    return "score_portex_scorer"


def _score_field(row: pd.Series, prefix: str, suffix: str) -> Any:
    """
    Retrieve a value from a pandas row using a composite score field key.
    
    Parameters:
        row (pd.Series): The row to read the value from.
        prefix (str): The score prefix to use (e.g., "score_portex_scorer").
        suffix (str): The field name suffix to append to the prefix (e.g., "metadata", "answer", "explanation").
    
    Returns:
        Any: The value at the column "{prefix}_{suffix}" if present, otherwise `None`.
    """
    return row.get(f"{prefix}_{suffix}")


def _event_call_cost(call: Any) -> float | None:
    """
    Extracts a numeric cost value from a model event call structure.
    
    Parameters:
        call (Any): The event call to inspect; may be None, a JSON string, or a dict. If a dict, the function looks for
            a "usage" object with a "cost" field under either the "response" or "request" keys.
    
    Returns:
        float | None: The cost as a float if present and convertible, `None` otherwise.
    """
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
    events["model_event_criterion_prompt"] = events["model_event_call"].apply(_event_call_criterion)
    return events


def _filter_events(
    events: pd.DataFrame,
    role: str,
    fallback_models: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of rows filtered by the specified event role or, when role information is unavailable, by a fallback set of model names.
    
    Parameters:
        events (pd.DataFrame): DataFrame containing model event rows; expected columns are `model_event_role` and/or `model_event_model`.
        role (str): The role value to filter `model_event_role` by.
        fallback_models (Iterable[str] | None): Iterable of model names to filter `model_event_model` by when `model_event_role` is not present or contains no values.
    
    Returns:
        pd.DataFrame: Rows where `model_event_role == role` if that column exists and has any non-null values;
        otherwise rows where `model_event_model` is in `fallback_models` if provided and the column exists;
        otherwise an empty DataFrame with the same columns as `events`.
    """
    if "model_event_role" in events.columns and events["model_event_role"].notna().any():
        return events[events["model_event_role"] == role]
    if fallback_models and "model_event_model" in events.columns:
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
    """
    Build a single-row evaluation-level DataFrame summarizing run, dataset, model, and aggregated usage.
    
    Processes task-level metadata and events to compute aggregate solver and scorer token usage and costs, summarize candidate and grader event latency/costs, and collect run- and dataset-level fields for the evaluation run.
    
    Parameters:
        log_row (pd.Series): Row from the evaluations log containing run-level fields (e.g., run_id, model, dataset_location).
        tasks_df (pd.DataFrame): DataFrame of task/sample rows for the run; used to extract per-task model usage and scoring metadata.
        events (pd.DataFrame): Events DataFrame containing model event records used to filter and summarize candidate and grader activity.
    
    Returns:
        pd.DataFrame: A one-row DataFrame with evaluation-level columns including run identifiers, dataset information, model configuration, aggregated solver and scorer usage (input/output/total tokens and cost), and summarized candidate/grader latency and cost.
    """
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
    score_prefix = _score_prefix(tasks_df)

    for _, row in tasks_df.iterrows():
        usage = _parse_json(row.get("model_usage"))
        if usage:
            usage_found = True
        if solver_model:
            solver_usage = _usage_for_models(usage, [solver_model])
            for key in eval_usage:
                if solver_usage.get(key) is not None:
                    eval_usage[key] = (eval_usage[key] or 0) + (solver_usage[key] or 0)

        metadata = _parse_json(_score_field(row, score_prefix, "metadata"))
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

    candidate_events = _filter_events(events, "candidate", [solver_model] if solver_model else None)
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
    """
    Builds a task-level DataFrame summarizing model runs, scores, and usage for each sample in tasks_df.
    
    Parameters:
        log_row (pd.Series): A single run record (from evals_df) providing run-level fields such as `model` and `dataset_location`.
        tasks_df (pd.DataFrame): The per-sample task dataframe (from samples_df) containing inputs, scores, and model usage columns.
        events (pd.DataFrame): Event records (from _build_events_df) used to summarize candidate and grader latency/cost per sample.
    
    Returns:
        pd.DataFrame: One row per sample/task containing run and sample identifiers, prompt and model response, scoring fields (grade, reasoning, PassThreshold, score), solver and scorer usage summaries (input/output/total tokens and cost), and latency estimates for candidate and grader events.
    """
    dataset_location = log_row.get("dataset_location")
    composite_id = _composite_id(dataset_location)
    solver_model = log_row.get("model")
    score_prefix = _score_prefix(tasks_df)

    rows: list[dict[str, Any]] = []
    for _, row in tasks_df.iterrows():
        metadata = _parse_json(_score_field(row, score_prefix, "metadata"))
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
                "model_response": _score_field(row, score_prefix, "answer"),
                "PassThreshold": metadata.get("pass_threshold"),
                "score": metadata.get("total_score"),
                "grade": row.get(score_prefix),
                "reasoning": _score_field(row, score_prefix, "explanation"),
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
    """
    Builds a criterion-level dataframe summarizing each criterion's grade and associated scorer usage for the given run.
    
    Parameters:
        log_row (pd.Series): The eval run record (one row from the evals dataframe) containing at least `dataset_location`.
        tasks_df (pd.DataFrame): Task-level dataframe containing samples, score metadata/predictions, and optional `model_usage`.
        events (pd.DataFrame): Events dataframe containing model event records (must include `sample_id`, `model_event_criterion_prompt`, and fields used by filtering/summarization).
    
    Returns:
        pd.DataFrame: A dataframe where each row represents a single criterion for a task in the run. Columns include:
            - run_id, composite_id, log, created, model, task_id, prompt, model_response
            - criterion_id, criterion_name, criterion_prompt, grader_type
            - criteria_points, criteria_awarded, criteria_passed, criteria_grade
            - scorer_usage_data_latency, scorer_usage_data_input_tokens, scorer_usage_data_output_tokens,
              scorer_usage_data_total_tokens, scorer_usage_data_cost
    """
    dataset_location = log_row.get("dataset_location")
    composite_id = _composite_id(dataset_location)
    score_prefix = _score_prefix(tasks_df)
    rows: list[dict[str, Any]] = []

    for _, row in tasks_df.iterrows():
        metadata = _parse_json(_score_field(row, score_prefix, "metadata"))
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
                    "model_response": _score_field(row, score_prefix, "answer"),
                    "criterion_id": criterion.get("criterion_id"),
                    "criterion_name": criterion.get("name"),
                    "criterion_prompt": prompt,
                    "grader_type": criterion.get("grader_type"),
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
    """
    Builds a dataframe with one row per judge for each criterion in each task of a single evaluation run.
    
    Parameters:
        log_row (pd.Series): A single evaluation run row (from the evals dataframe); used for run- and dataset-level fields.
        tasks_df (pd.DataFrame): Task/sample rows for the run containing score metadata, model responses, and model usage.
        events (pd.DataFrame): Event-level records (as produced by _build_events_df) used to locate judge call events and summarize latency/cost.
    
    Returns:
        pd.DataFrame: A dataframe where each row represents a judge's judgement for a specific criterion on a task. Columns include run identifiers (run_id, composite_id, log, created), model/task context (model, task_id, prompt, model_response), criterion fields (criterion_id, criterion_name, criterion_prompt, grader_type, criteria_points), judge fields (judge_name, judge_awarded, judge_passed, judge_grade, judge_reasoning), and aggregated judge usage/summarized metrics (scorer_usage_data_latency, scorer_usage_data_input_tokens, scorer_usage_data_output_tokens, scorer_usage_data_total_tokens, scorer_usage_data_cost).
    """
    dataset_location = log_row.get("dataset_location")
    composite_id = _composite_id(dataset_location)
    score_prefix = _score_prefix(tasks_df)
    rows: list[dict[str, Any]] = []

    for _, row in tasks_df.iterrows():
        metadata = _parse_json(_score_field(row, score_prefix, "metadata"))
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
                        "model_response": _score_field(row, score_prefix, "answer"),
                        "criterion_id": criterion.get("criterion_id"),
                        "criterion_name": criterion.get("name"),
                        "criterion_prompt": criterion.get("prompt"),
                        "grader_type": criterion.get("grader_type"),
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
