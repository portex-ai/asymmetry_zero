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
    """
    Extracts the agent model value from a list of Harbor CLI extra arguments.
    
    Parameters:
        extra_args (list[str] | None): Additional command-line arguments passed to Harbor.
    
    Returns:
        str | None: The value following a `--model` flag if present, `None` otherwise.
    """
    if not extra_args:
        return None
    for idx, arg in enumerate(extra_args):
        if arg == "--model" and idx + 1 < len(extra_args):
            return extra_args[idx + 1]
    return None


def _default_api_key_env(provider: str) -> str | None:
    """
    Map a provider identifier to the default environment variable name used for its API key.
    
    Parameters:
        provider (str): Provider identifier (e.g., "openrouter", "openai", "anthropic").
    
    Returns:
        str | None: The environment variable name for the provider's API key, or `None` if the provider is unrecognized.
    """
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
    """
    Materialize API key secrets for judge model specifications so they can be embedded in child process envs or config files.
    
    Parameters:
        specs (list[ModelSpec]): Model specifications for judges to convert into concrete model configurations.
        env (dict[str, str]): Environment mapping used to look up API key values by environment-variable name.
    
    Returns:
        list[dict[str, object]]: List of model configuration dictionaries where each config's `api_key` is populated from the spec or the provided environment (when available) and `api_key_env` cleared when the key was inlined.
    """
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
    """
    Run a Harbor evaluation using task files under task_root and produce structured run artifacts.
    
    Runs the Harbor CLI to execute tasks found in task_root/datasets, optionally injecting judge model configurations and extra Harbor CLI arguments, then collects and returns the produced reports, rewards, and artifact paths as a HarborRunResult.
    
    Parameters:
        task_root (str): Path to the task root directory containing a `datasets` subdirectory and where output will be written.
        judges (list[ModelSpec] | None): Optional judge specifications to materialize into child process environment variables for Harbor.
        n_concurrent (int | None): Optional concurrency limit passed to Harbor.
        env (str | None): Optional Harbor environment name to pass through to Harbor.
        extra_args (list[str] | None): Additional CLI arguments forwarded to Harbor (e.g., ["--model", "gpt-4"]).
        overwrite (bool): If True, allow overwriting an existing jobs directory for the new run; otherwise raise an error if it exists.
    
    Returns:
        HarborRunResult: Immutable container with run metadata, report paths, rewards payload, and artifact paths.
    
    Raises:
        ModuleNotFoundError: If the Harbor CLI package is not installed in the environment.
        FileNotFoundError: If the expected `datasets` directory is missing under task_root.
        ValueError: If the target jobs directory already exists and overwrite is False.
        subprocess.CalledProcessError: If the Harbor process exits with a non-zero status.
    """
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
    """
    Convert a HarborRunResult into an AgentEvalResults object with all file paths made absolute.
    
    Preserves datasets_dir, jobs_dir, reports, rewards, rewards_path, training_data_path, run_id, and output_dir from the input result.
    
    Returns:
        AgentEvalResults: The API-facing results object with absolute paths.
    """
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
