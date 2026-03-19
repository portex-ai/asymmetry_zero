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
from typing import Any

from portex_eval.providers import (
    ModelConfig,
    ModelSpec,
    model_config_from_spec,
    model_config_to_dict,
)
from portex_eval.types import AgentEvalResults

PORTEX_MULTIMODAL_AGENT_NAME = "portex-multimodal"
PORTEX_MULTIMODAL_AGENT_IMPORT_PATH = (
    "portex_eval.benchmark.harbor.agent:PortexMultimodalAgent"
)
MODEL_CONFIGS_PATH = Path(__file__).resolve().parent / "resources" / "model_configs.json"


@dataclass(frozen=True)
class HarborRunResult:
    run_id: str
    output_dir: str
    datasets_dir: str
    jobs_dir: str


def _extract_agent_model(extra_args: list[str] | None) -> str | None:
    if not extra_args:
        return None
    for idx, arg in enumerate(extra_args):
        if arg in {"--model", "-m"} and idx + 1 < len(extra_args):
            return extra_args[idx + 1]
        if arg.startswith("--model="):
            return arg.split("=", 1)[1]
    return None


def _default_api_key_env(provider: str) -> str | None:
    if provider == "openrouter":
        return "OPENROUTER_API_KEY"
    if provider == "openai":
        return "OPENAI_API_KEY"
    if provider == "anthropic":
        return "ANTHROPIC_API_KEY"
    return None


def _load_model_configs(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(model_name): dict(model_info)
        for model_name, model_info in data.items()
        if isinstance(model_info, dict)
    }


def _model_info_for_model_name(model_name: str) -> dict[str, Any] | None:
    model_info = _load_model_configs(MODEL_CONFIGS_PATH).get(model_name)
    if not isinstance(model_info, dict):
        return None

    normalized = dict(model_info)
    if model_name.startswith("openrouter/"):
        canonical = model_name.split("/", 1)[1]
        provider = canonical.split("/", 1)[0] if "/" in canonical else canonical
        if provider.lower() in {"anthropic", "google", "openai"}:
            normalized["litellm_provider"] = provider.lower()
    return normalized


def _has_agent_kwarg(extra_args: list[str], key: str) -> bool:
    prefix = f"{key}="
    for idx, arg in enumerate(extra_args):
        if arg in {"--ak", "--agent-kwarg"} and idx + 1 < len(extra_args):
            if extra_args[idx + 1].startswith(prefix):
                return True
        if arg.startswith("--ak=") and arg[len("--ak=") :].startswith(prefix):
            return True
        if arg.startswith("--agent-kwarg=") and arg[len("--agent-kwarg=") :].startswith(prefix):
            return True
    return False


def _rewrite_portex_multimodal_agent(extra_args: list[str]) -> list[str]:
    if "--agent-import-path" in extra_args:
        return extra_args

    rewritten: list[str] = []
    idx = 0
    while idx < len(extra_args):
        arg = extra_args[idx]
        if arg in {"--agent", "-a"} and idx + 1 < len(extra_args):
            agent_name = extra_args[idx + 1]
            if agent_name == PORTEX_MULTIMODAL_AGENT_NAME:
                rewritten.extend(
                    [
                        "--agent-import-path",
                        PORTEX_MULTIMODAL_AGENT_IMPORT_PATH,
                    ]
                )
            else:
                rewritten.extend([arg, agent_name])
            idx += 2
            continue
        if arg.startswith("--agent="):
            agent_name = arg.split("=", 1)[1]
            if agent_name == PORTEX_MULTIMODAL_AGENT_NAME:
                rewritten.extend(
                    [
                        "--agent-import-path",
                        PORTEX_MULTIMODAL_AGENT_IMPORT_PATH,
                    ]
                )
            else:
                rewritten.append(arg)
            idx += 1
            continue
        rewritten.append(arg)
        idx += 1
    return rewritten


def _maybe_inject_model_info(extra_args: list[str]) -> list[str]:
    if not extra_args or _has_agent_kwarg(extra_args, "model_info"):
        return extra_args

    model_name = _extract_agent_model(extra_args)
    if not model_name:
        return extra_args

    model_info = _model_info_for_model_name(model_name)
    if model_info is None:
        return extra_args

    return extra_args + [
        "--ak",
        f"model_info={json.dumps(model_info, separators=(',', ':'))}",
    ]


def _normalize_harbor_extra_args(extra_args: list[str] | None) -> list[str]:
    normalized = list(extra_args or [])
    normalized = _rewrite_portex_multimodal_agent(normalized)
    normalized = _maybe_inject_model_info(normalized)
    return normalized


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


def _scrub_verifier_env_placeholders(task_root: Path) -> None:
    for task_toml in task_root.glob("datasets/*/task.toml"):
        try:
            original = task_toml.read_text(encoding="utf-8")
        except OSError:
            continue
        updated_lines = [
            line
            for line in original.splitlines()
            if "PORTEX_JUDGE_MODELS" not in line and "PORTEX_JUDGE_CONFIGS" not in line
        ]
        updated = "\n".join(updated_lines)
        if original.endswith("\n"):
            updated += "\n"
        if updated != original:
            task_toml.write_text(updated, encoding="utf-8")


def _apply_judge_overrides(task_root: Path, judges: list[ModelSpec] | None, env: dict[str, str]) -> None:
    if not judges:
        return

    configs = [model_config_from_spec(spec) for spec in judges]
    normalized_models = [f"{config.provider}:{config.model}" for config in configs]
    materialized_configs = _materialize_judge_config_secrets(judges, env)

    for task_config_path in task_root.glob("datasets/*/tests/task_config.json"):
        try:
            task_config = json.loads(task_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(task_config, dict):
            continue
        task_config["judge_models"] = normalized_models
        task_config["judge_configs"] = materialized_configs
        task_config_path.write_text(json.dumps(task_config, indent=2), encoding="utf-8")


def run_harbor_tasks(
    *,
    task_root: str,
    judges: list[ModelSpec] | None = None,
    n_concurrent: int | None = None,
    env: str | None = None,
    extra_args: list[str] | None = None,
    overwrite: bool = False,
) -> HarborRunResult:
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
    normalized_extra_args = _normalize_harbor_extra_args(extra_args)

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
    if normalized_extra_args:
        cmd.extend(normalized_extra_args)

    child_env = os.environ.copy()
    _scrub_verifier_env_placeholders(root)
    _apply_judge_overrides(root, judges, child_env)
    if judges:
        configs = [model_config_from_spec(spec) for spec in judges]
        child_env["PORTEX_JUDGE_CONFIGS"] = json.dumps(
            _materialize_judge_config_secrets(judges, child_env)
        )
        child_env["PORTEX_JUDGE_MODELS"] = ",".join(
            f"{config.provider}:{config.model}" for config in configs
        )

    subprocess.run(cmd, check=True, env=child_env)

    return HarborRunResult(
        run_id=run_id,
        output_dir=str(root),
        datasets_dir=str(datasets_dir),
        jobs_dir=str(jobs_dir),
    )


def harbor_run_result_to_api(result: HarborRunResult) -> AgentEvalResults:
    return AgentEvalResults(
        datasets_dir=result.datasets_dir,
        jobs_dir=result.jobs_dir,
        run_id=result.run_id,
        output_dir=result.output_dir,
    ).with_absolute_paths()
