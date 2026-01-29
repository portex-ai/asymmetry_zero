from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Any


def _write_json(path: str, obj: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _list_files_recursive(root: str) -> set[str]:
    files: set[str] = set()
    if not os.path.exists(root):
        return files
    for base, _, fns in os.walk(root):
        for fn in fns:
            files.add(os.path.join(base, fn))
    return files


def _run_inspect_eval(cmd: list[str], env: dict[str, str]) -> None:
    subprocess.run(cmd, check=True, env=env)


def run_inspect_eval(
    log_dir: str,
    report_path: str,
    data_dir: str,
    model: str,
    task_spec: str,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    before = _list_files_recursive(log_dir)

    env = dict(os.environ)
    env["PORTEX_EVAL_DATA_DIR"] = data_dir
    env["DATA_DIR"] = data_dir
    if extra_env:
        env.update(extra_env)

    cmd = [
        "inspect",
        "eval",
        task_spec,
        "--model",
        model,
        "--log-dir",
        log_dir,
    ]

    _run_inspect_eval(cmd, env=env)

    after = _list_files_recursive(log_dir)
    report = {
        "log_dir": log_dir,
        "log_files": sorted(after - before),
    }
    _write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Inspect eval for Portex Eval")
    parser.add_argument(
        "--report-path",
        default=os.getenv("REPORT_PATH", "./runs/report.json"),
    )
    parser.add_argument(
        "--log-dir",
        default=os.getenv("LOG_DIR", "./runs"),
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("DATA_DIR", "./bundle"),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL", "openrouter/google/gemini-2.5-flash"),
    )
    parser.add_argument(
        "--task-spec",
        default=os.getenv("TASK_SPEC", None),
    )
    args = parser.parse_args()

    if not args.task_spec:
        raise ValueError("--task-spec is required when using this module directly")

    run_inspect_eval(
        log_dir=args.log_dir,
        report_path=args.report_path,
        data_dir=args.data_dir,
        model=args.model,
        task_spec=args.task_spec,
    )


if __name__ == "__main__":
    main()
