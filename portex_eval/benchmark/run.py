from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from portex_eval.benchmark.inspect.main import run_inspect_eval

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
    if task_spec is None:
        return f"{Path(__file__).resolve().parent / 'inspect' / 'eval.py'}@portex_qa_eval"
    if task_spec != "portex_qa_eval":
        raise ValueError(f"Task spec {task_spec} not supported")
    return f"{Path(__file__).resolve().parent / 'inspect' / 'eval.py'}@{task_spec}"


def benchmark_one(
    *,
    bundle_dir: str,
    index_root: str | None = None,
    eval_runs_root: str | None = None,
    model_endpoint: str,
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
        model_endpoint: Model identifier (e.g., "openrouter/google/gemini-2.5-flash")
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

    task_spec_resolved = _resolve_task_spec(task_spec)
    run_id = datetime.utcnow().isoformat(timespec="seconds").replace(":", "-") + "Z"

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
    report = run_inspect_eval(
        log_dir=logs_dir,
        report_path=report_path,
        data_dir=bundle_dir,
        model=model_endpoint,
        task_spec=task_spec_resolved,
        max_samples=max_samples,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
    )

    eval_log = _pick_eval_log(report, logs_dir)

    bundle_manifest = _read_manifest(os.path.join(bundle_dir, "manifest.json"))

    manifest_payload: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "index_root": index_root,
        "eval_runs_root": eval_runs_root,
        "bundle_dir": bundle_dir,
        "bundle_relpath": rel,
        "model_endpoint": model_endpoint,
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
    models: list[str],
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
        models: List of model endpoints
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
                    model_endpoint=model,
                    task_spec=task_spec,
                    max_samples=max_samples,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    overwrite=overwrite,
                )
            )
    return BenchmarkMatrixResult(results=results)
