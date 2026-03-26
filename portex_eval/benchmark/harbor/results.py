"""Artifact generation for Harbor-backed agent eval runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from portex_eval.rewards import build_rewards, write_rewards


def _detail_files(jobs_dir: str) -> list[Path]:
    return sorted(Path(jobs_dir).rglob("portex_detail.json"))


def _load_details(jobs_dir: str) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for path in _detail_files(jobs_dir):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload["_detail_path"] = str(path)
            details.append(payload)
    return details


def _task_scores(details: list[dict[str, Any]]) -> list[tuple[str, float]]:
    scores: list[tuple[str, float]] = []
    for detail in details:
        task_id = detail.get("task_id")
        if isinstance(task_id, str) and task_id:
            scores.append((task_id, float(detail.get("total_score_raw", 0.0) or 0.0)))
    return scores


def _harbor_training_data(
    details: list[dict[str, Any]],
    task_scores: list[tuple[str, float]],
    *,
    jobs_dir: str,
    agent_model: str | None,
) -> dict[str, Any]:
    reward_by_task = {task_id: score for task_id, score in task_scores}
    records: list[dict[str, Any]] = []
    for detail in details:
        task_id = detail.get("task_id")
        question = str(detail.get("question") or "")
        reference_file = str(detail.get("reference_file") or "")
        content = [{"type": "text", "text": question}]
        if reference_file:
            suffix = Path(reference_file).suffix.lower()
            content.append(
                {
                    "type": "image" if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"} else "document",
                    "image" if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"} else "document": reference_file,
                }
            )
        prompt_text = f"user:\n{question}"
        if reference_file:
            prompt_text += "\n[image]" if Path(reference_file).suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
                ".gif",
            } else "\n[document]"
        records.append(
            {
                "task_id": task_id,
                "sample_id": task_id,
                "epoch": 1,
                "model": agent_model,
                "reward": reward_by_task.get(str(task_id)),
                "reference_file": reference_file,
                "prompt_messages": [{"role": "user", "content": content}],
                "prompt_text": prompt_text,
                "completion": detail.get("submission"),
                "prompt_token_ids": None,
                "completion_token_ids": None,
                "completion_logprobs": None,
            }
        )

    return {
        "format": "portex-rl-training-data",
        "version": 1,
        "source": {
            "jobs_dir": str(Path(jobs_dir).resolve()),
        },
        "records": records,
    }


def write_harbor_artifacts(
    *,
    jobs_dir: str,
    output_dir: str,
    run_id: str,
    datasets_dir: str,
    agent_model: str | None,
    harbor_args: list[str],
) -> tuple[tuple[str, str, str, str], Any, str, str]:
    output_root = Path(output_dir)
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    details = _load_details(jobs_dir)
    task_scores = _task_scores(details)

    eval_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    criterion_rows: list[dict[str, Any]] = []
    judgement_rows: list[dict[str, Any]] = []

    avg_score = (
        sum(float(detail.get("total_score_raw", 0.0) or 0.0) for detail in details) / len(details)
        if details
        else 0.0
    )

    eval_rows.append(
        {
            "run_id": run_id,
            "composite_id": output_root.name,
            "log": "",
            "created": run_id.replace("-", ":", 2),
            "packages": "",
            "task_file": "",
            "model": agent_model,
            "model_base_url": None,
            "model_arguments": json.dumps(harbor_args),
            "model_generation_config": None,
            "dataset_location": datasets_dir,
            "dataset_samples": len(details),
            "dataset_sample_ids": json.dumps([detail.get("task_id") for detail in details]),
            "epochs": 1,
            "status": "success",
            "total_samples": len(details),
            "completed_samples": len(details),
            "scored_headline_names": "reward",
            "scored_headline_metrics": "mean",
            "score_headline_value": avg_score,
            "score_headline_stderr": None,
            "solver_usage_data_latency": None,
            "solver_usage_data_input_tokens": None,
            "solver_usage_data_output_tokens": None,
            "solver_usage_data_total_tokens": None,
            "solver_usage_data_cost": None,
            "scorer_usage_data_latency": None,
            "scorer_usage_data_input_tokens": None,
            "scorer_usage_data_output_tokens": None,
            "scorer_usage_data_total_tokens": None,
            "scorer_usage_data_cost": None,
        }
    )

    for detail in details:
        task_rows.append(
            {
                "run_id": run_id,
                "composite_id": output_root.name,
                "log": "",
                "created": run_id.replace("-", ":", 2),
                "model": agent_model,
                "task_id": detail.get("task_id"),
                "prompt": detail.get("question"),
                "model_response": detail.get("submission"),
                "PassThreshold": detail.get("pass_threshold"),
                "score": detail.get("total_score_raw"),
                "grade": detail.get("grade"),
                "reasoning": detail.get("error"),
                "solver_usage_data_latency": None,
                "solver_usage_data_input_tokens": None,
                "solver_usage_data_output_tokens": None,
                "solver_usage_data_total_tokens": None,
                "solver_usage_data_cost": None,
                "scorer_usage_data_latency": None,
                "scorer_usage_data_input_tokens": None,
                "scorer_usage_data_output_tokens": None,
                "scorer_usage_data_total_tokens": None,
                "scorer_usage_data_cost": None,
            }
        )
        for criterion in detail.get("criteria_results", []) or []:
            criterion_rows.append(
                {
                    "run_id": run_id,
                    "composite_id": output_root.name,
                    "log": "",
                    "created": run_id.replace("-", ":", 2),
                    "model": agent_model,
                    "task_id": detail.get("task_id"),
                    "prompt": detail.get("question"),
                    "model_response": detail.get("submission"),
                    "criterion_id": criterion.get("criterion_id"),
                    "criterion_name": criterion.get("name"),
                    "criterion_prompt": criterion.get("prompt") or criterion.get("semanticPrompt"),
                    "grader_type": criterion.get("grader_type"),
                    "criteria_points": criterion.get("weight"),
                    "criteria_awarded": criterion.get("awarded"),
                    "criteria_passed": criterion.get("passed"),
                    "criteria_grade": criterion.get("grade"),
                    "scorer_usage_data_latency": None,
                    "scorer_usage_data_input_tokens": None,
                    "scorer_usage_data_output_tokens": None,
                    "scorer_usage_data_total_tokens": None,
                    "scorer_usage_data_cost": None,
                }
            )
            for judge in criterion.get("judges", []) or []:
                judgement_rows.append(
                    {
                        "run_id": run_id,
                        "composite_id": output_root.name,
                        "log": "",
                        "created": run_id.replace("-", ":", 2),
                        "model": agent_model,
                        "task_id": detail.get("task_id"),
                        "prompt": detail.get("question"),
                        "model_response": detail.get("submission"),
                        "criterion_id": criterion.get("criterion_id"),
                        "criterion_name": criterion.get("name"),
                        "criterion_prompt": criterion.get("prompt") or criterion.get("semanticPrompt"),
                        "grader_type": criterion.get("grader_type"),
                        "criteria_points": criterion.get("weight"),
                        "judge_name": judge.get("model"),
                        "judge_awarded": judge.get("awarded"),
                        "judge_passed": judge.get("passed"),
                        "judge_grade": judge.get("grade"),
                        "judge_reasoning": judge.get("explanation"),
                        "scorer_usage_data_latency": None,
                        "scorer_usage_data_input_tokens": None,
                        "scorer_usage_data_output_tokens": None,
                        "scorer_usage_data_total_tokens": None,
                        "scorer_usage_data_cost": None,
                    }
                )

    eval_level = reports_dir / "eval_level.csv"
    task_level = reports_dir / "task_level.csv"
    criterion_level = reports_dir / "criterion_level.csv"
    judgement_level = reports_dir / "judgement_level.csv"
    pd.DataFrame(eval_rows).to_csv(eval_level, index=False)
    pd.DataFrame(task_rows).to_csv(task_level, index=False)
    pd.DataFrame(criterion_rows).to_csv(criterion_level, index=False)
    pd.DataFrame(judgement_rows).to_csv(judgement_level, index=False)

    rewards_path = write_rewards(task_scores, str(output_root / "rl_rewards.json"))
    rewards_payload = build_rewards(task_scores)
    training_data_path = output_root / "rl_training_data.json"
    training_data_path.write_text(
        json.dumps(
            _harbor_training_data(
                details,
                task_scores,
                jobs_dir=jobs_dir,
                agent_model=agent_model,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    return (
        (str(eval_level), str(task_level), str(criterion_level), str(judgement_level)),
        rewards_payload,
        rewards_path,
        str(training_data_path.resolve()),
    )
