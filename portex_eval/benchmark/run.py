from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from portex_eval.benchmark.inspect.main import run_inspect_eval
from portex_eval.providers import ModelConfig, ModelSpec, model_config_from_spec, model_config_to_dict

# Local defaults (simplified from portex_eval.config)
DEFAULT_EVAL_BUNDLES_ROOT = os.getenv("PORTEX_EVAL_BUNDLES_ROOT", "./bundles")
DEFAULT_EVAL_RUNS_ROOT = os.getenv("PORTEX_EVAL_RUNS_ROOT", "./eval_runs")


@dataclass
class BenchmarkResult:
    run_id: str
    output_dir: str
    eval_log: str
    report_path: str


@dataclass
class BenchmarkMatrixResult:
    results: list[BenchmarkResult]


def _read_manifest(path: str) -> dict[str, Any]:
    """Read a manifest JSON file, returning empty dict if not found."""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _write_manifest(path: str, payload: dict[str, Any]) -> None:
    """Write a manifest JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _pick_eval_log(report: dict[str, Any], log_dir: str) -> str:
    files = report.get("log_files") if isinstance(report, dict) else None
    if isinstance(files, list) and files:
        eval_logs = [f for f in files if isinstance(f, str) and f.endswith(".eval")]
        if len(eval_logs) == 1:
            return eval_logs[0]
        if eval_logs:
            return sorted(eval_logs)[-1]
    candidates = list(Path(log_dir).glob("*.eval"))
    if not candidates:
        raise ValueError(f"No .eval logs found under {log_dir}")
    return str(sorted(candidates)[-1])


def _resolve_task_spec(task_spec: str | None) -> str:
    return _resolve_task_spec_for_mode(task_spec, use_providers=False)


def _resolve_task_spec_for_mode(task_spec: str | None, *, use_providers: bool) -> str:
    task_name = task_spec or (
        "portex_qa_eval_with_providers" if use_providers else "portex_qa_eval"
    )
    if task_name not in {"portex_qa_eval", "portex_qa_eval_with_providers"}:
        raise ValueError(f"Task spec {task_spec} not supported")
    return f"{Path(__file__).resolve().parent / 'inspect' / 'eval.py'}@{task_name}"


def _spec_needs_provider_runtime(config: ModelConfig) -> bool:
    return (
        config.provider != "openrouter"
        or config.base_url is not None
        or config.api_key is not None
        or config.api_key_env is not None
        or bool(config.headers)
        or bool(config.options)
    )


def _use_provider_runtime(candidate: ModelConfig, judges: list[ModelConfig]) -> bool:
    return _spec_needs_provider_runtime(candidate) or any(
        _spec_needs_provider_runtime(judge) for judge in judges
    )


def _inspect_model_name(config: ModelConfig) -> str:
    if config.provider == "openrouter":
        return f"openrouter/{config.model}"
    if config.provider in {"openai", "openai_compatible", "openai-compatible", "vllm", "custom"}:
        return f"openai/{config.model}"
    if config.provider == "anthropic":
        return f"anthropic/{config.model}"
    return f"openai/{config.model}"


def _provider_env(
    candidate: ModelConfig,
    judges: list[ModelConfig],
    *,
    logprobs: bool,
    top_logprobs: int | None,
) -> dict[str, str]:
    env = {
        "PORTEX_CANDIDATE_MODEL": candidate.model_string,
        "PORTEX_CANDIDATE_CONFIG": json.dumps(model_config_to_dict(candidate)),
        "PORTEX_JUDGE_MODELS": ",".join(judge.model_string for judge in judges),
        "PORTEX_JUDGE_CONFIGS": json.dumps([model_config_to_dict(judge) for judge in judges]),
    }
    if logprobs:
        env["PORTEX_LOGPROBS"] = "true"
    if top_logprobs is not None:
        env["PORTEX_TOP_LOGPROBS"] = str(top_logprobs)
    return env


def benchmark_one(
    *,
    bundle_dir: str,
    index_root: str | None = None,
    eval_runs_root: str | None = None,
    candidate_spec: ModelSpec,
    judge_specs: list[ModelSpec],
    task_spec: str | None = None,
    max_samples: int | None = None,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    overwrite: bool = False,
) -> BenchmarkResult:
    """Run a single benchmark evaluation.

    Args:
        bundle_dir: Path to the bundle directory containing tasks.json and answers.json
        index_root: Root directory for bundles (used for relative path calculation)
        eval_runs_root: Root directory for eval run outputs
        candidate_spec: Candidate model spec.
        judge_specs: Judge model specs.
        task_spec: Task specification (defaults to portex_qa_eval)
        max_samples: Maximum number of dataset samples to run in parallel.
        logprobs: Whether to request completion logprobs from the candidate model.
        top_logprobs: Number of top logprob alternatives to request per completion token.
        overwrite: If True, allow overwriting existing output directories.
            Defaults to False to prevent accidental data loss.

    Returns:
        BenchmarkResult with run details

    Raises:
        ValueError: If output directory already exists and overwrite is False.
    """
    index_root = index_root or DEFAULT_EVAL_BUNDLES_ROOT
    eval_runs_root = eval_runs_root or DEFAULT_EVAL_RUNS_ROOT

    if not os.path.isdir(bundle_dir):
        raise ValueError(f"Bundle directory not found: {bundle_dir}")

    candidate_config = model_config_from_spec(candidate_spec)
    judge_configs = [model_config_from_spec(spec) for spec in judge_specs]
    use_providers = _use_provider_runtime(candidate_config, judge_configs)
    task_spec_resolved = _resolve_task_spec_for_mode(task_spec, use_providers=use_providers)
    run_id = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(":", "-")

    try:
        rel = str(Path(bundle_dir).resolve().relative_to(Path(index_root).resolve()))
    except Exception:
        rel = os.path.basename(bundle_dir.rstrip("/"))
    output_dir = os.path.join(eval_runs_root, rel, run_id)

    if os.path.exists(output_dir) and not overwrite:
        raise ValueError(
            f"Output directory already exists: {output_dir}. "
            "Use overwrite=True to allow overwriting."
        )

    logs_dir = os.path.join(output_dir, "logs")
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "manifest.json")
    extra_env = (
        _provider_env(
            candidate_config,
            judge_configs,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        if use_providers
        else {"PORTEX_JUDGE_MODELS": ",".join(_inspect_model_name(j) for j in judge_configs)}
    )
    report = run_inspect_eval(
        log_dir=logs_dir,
        report_path=report_path,
        data_dir=bundle_dir,
        model=None if use_providers else _inspect_model_name(candidate_config),
        task_spec=task_spec_resolved,
        max_samples=max_samples,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        extra_env=extra_env,
    )

    eval_log = _pick_eval_log(report, logs_dir)

    bundle_manifest = _read_manifest(os.path.join(bundle_dir, "manifest.json"))

    manifest_payload: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "index_root": index_root,
        "eval_runs_root": eval_runs_root,
        "bundle_dir": bundle_dir,
        "bundle_relpath": rel,
        "candidate_spec": model_config_to_dict(candidate_config),
        "judge_specs": [model_config_to_dict(model) for model in judge_configs],
        "model_endpoint": _inspect_model_name(candidate_config),
        "use_providers": use_providers,
        "task_spec": task_spec_resolved,
        "eval_log": eval_log,
        "report_path": report_path,
        "bundle_manifest": bundle_manifest,
    }

    # Enrich manifest.json with identifiers so it can be linked back to the bundle/version.
    try:
        if os.path.isfile(report_path):
            with open(report_path, encoding="utf-8") as f:
                report_obj = json.load(f)
        else:
            report_obj = {}
    except Exception:
        report_obj = {}

    # Keep inspect-produced fields, but add everything we store in manifest.json.
    report_obj.update(manifest_payload)
    report_obj["bundle_name"] = os.path.basename(bundle_dir.rstrip("/"))

    try:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_obj, f, indent=2)
    except OSError:
        pass

    _write_manifest(report_path, report_obj)

    return BenchmarkResult(
        run_id=run_id, output_dir=output_dir, eval_log=eval_log, report_path=report_path
    )


def benchmark_matrix(
    bundle_dirs: list[str],
    models: list[ModelSpec],
    judge_specs: list[ModelSpec],
    index_root: str | None = None,
    eval_runs_root: str | None = None,
    task_spec: str | None = None,
    max_samples: int | None = None,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    overwrite: bool = False,
) -> BenchmarkMatrixResult:
    """Run benchmarks across multiple bundles and models.

    Args:
        bundle_dirs: List of bundle directory paths
        models: List of candidate model specs
        judge_specs: Shared judge model specs
        index_root: Root directory for bundles
        eval_runs_root: Root directory for eval run outputs
        task_spec: Task specification
        max_samples: Maximum number of dataset samples to run in parallel.
        logprobs: Whether to request completion logprobs from the candidate model.
        top_logprobs: Number of top logprob alternatives to request per completion token.
        overwrite: If True, allow overwriting existing output directories.

    Returns:
        BenchmarkMatrixResult containing all run results
    """
    results: list[BenchmarkResult] = []
    for bundle_dir in bundle_dirs:
        for model in models:
            results.append(
                benchmark_one(
                    bundle_dir=bundle_dir,
                    index_root=index_root,
                    eval_runs_root=eval_runs_root,
                    candidate_spec=model,
                    judge_specs=judge_specs,
                    task_spec=task_spec,
                    max_samples=max_samples,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    overwrite=overwrite,
                )
            )
    return BenchmarkMatrixResult(results=results)
