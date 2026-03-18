"""Wrapper for running Harbor-backed agent evals."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from portex_eval.providers import (
    ModelConfig,
    ModelSpec,
    model_config_from_spec,
    model_config_to_dict,
)
from portex_eval.types import AgentEvalResults, ReportPaths, Rewards


@dataclass(frozen=True)
class HarborRunResult:
    run_id: str
    output_dir: str
    datasets_dir: str
    jobs_dir: str
    reports: ReportPaths
    rewards: Rewards
    rewards_path: str
    training_data_path: str


def _extract_agent_model(extra_args: list[str] | None) -> str | None:
    if not extra_args:
        return None
    for idx, arg in enumerate(extra_args):
        if arg == "--model" and idx + 1 < len(extra_args):
            return extra_args[idx + 1]
    return None


def _default_api_key_env(provider: str) -> str | None:
    if provider == "openrouter":
        return "OPENROUTER_API_KEY"
    if provider == "openai":
        return "OPENAI_API_KEY"
    if provider == "anthropic":
        return "ANTHROPIC_API_KEY"
    return None


def _materialize_judge_config_secrets(
    specs: list[ModelSpec],
    env: dict[str, str],
) -> list[dict[str, object]]:
    materialized: list[dict[str, object]] = []
    for spec in specs:
        config = model_config_from_spec(spec)
        api_key = config.api_key
        api_key_env = config.api_key_env or _default_api_key_env(config.provider)
        if api_key is None and api_key_env:
            api_key = env.get(api_key_env)

        if api_key is not None:
            config = replace(config, api_key=api_key, api_key_env=None)

        materialized.append(model_config_to_dict(config))
    return materialized


def run_harbor_tasks(
    *,
    task_root: str,
    judges: list[ModelSpec] | None = None,
    n_concurrent: int | None = None,
    env: str | None = None,
    extra_args: list[str] | None = None,
    overwrite: bool = False,
) -> HarborRunResult:
    from portex_eval.benchmark.harbor.results import write_harbor_artifacts

    try:
        harbor_spec = importlib.util.find_spec("harbor.cli.main")
    except ModuleNotFoundError:
        harbor_spec = None
    if harbor_spec is None:
        raise ModuleNotFoundError(
            "Harbor is not installed in this environment. "
            "Install the known-good Harbor stack with "
            "`uv sync --group harbor` or `pip install 'portex-eval[harbor]'`."
        )

    root = Path(task_root).expanduser().resolve()
    datasets_dir = root / "datasets"
    if not datasets_dir.is_dir():
        raise FileNotFoundError(f"Harbor datasets directory not found: {datasets_dir}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jobs_dir = root / "jobs" / run_id
    if jobs_dir.exists() and not overwrite:
        raise ValueError(
            f"Jobs directory already exists: {jobs_dir}. Use overwrite=True to allow overwriting."
        )
    jobs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "harbor.cli.main",
        "run",
        "-p",
        str(datasets_dir),
        "-o",
        str(jobs_dir),
    ]
    if n_concurrent is not None:
        cmd.extend(["--n-concurrent", str(int(n_concurrent))])
    if env is not None:
        cmd.extend(["--env", env])
    if extra_args:
        cmd.extend(extra_args)

    child_env = os.environ.copy()
    if judges:
        configs = [model_config_from_spec(spec) for spec in judges]
        child_env["PORTEX_JUDGE_CONFIGS"] = json.dumps(
            _materialize_judge_config_secrets(judges, child_env)
        )
        child_env["PORTEX_JUDGE_MODELS"] = ",".join(
            f"{config.provider}:{config.model}" for config in configs
        )

    subprocess.run(cmd, check=True, env=child_env)

    agent_model = _extract_agent_model(extra_args)
    report_paths, rewards_payload, rewards_path, training_data_path = write_harbor_artifacts(
        jobs_dir=str(jobs_dir),
        output_dir=str(root),
        run_id=run_id,
        datasets_dir=str(datasets_dir),
        agent_model=agent_model,
        harbor_args=extra_args or [],
    )

    eval_level, task_level, criterion_level, judgement_level = report_paths
    return HarborRunResult(
        run_id=run_id,
        output_dir=str(root),
        datasets_dir=str(datasets_dir),
        jobs_dir=str(jobs_dir),
        reports=ReportPaths(
            eval_level=eval_level,
            task_level=task_level,
            criterion_level=criterion_level,
            judgement_level=judgement_level,
        ),
        rewards=rewards_payload,
        rewards_path=rewards_path,
        training_data_path=training_data_path,
    )


def harbor_run_result_to_api(result: HarborRunResult) -> AgentEvalResults:
    return AgentEvalResults(
        datasets_dir=result.datasets_dir,
        jobs_dir=result.jobs_dir,
        reports=result.reports,
        rewards=result.rewards,
        rewards_path=result.rewards_path,
        training_data_path=result.training_data_path,
        run_id=result.run_id,
        output_dir=result.output_dir,
    ).with_absolute_paths()
