"""Convert Portex bundles into Harbor task directories."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from portex_eval.grading import DEFAULT_AGENT_JUDGE_MODELS
from portex_eval.types import AgentEvalBundle

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
DEFAULT_BASE_IMAGE = "ubuntu:24.04"
DEFAULT_AGENT_TIMEOUT = 3600
DEFAULT_VERIFIER_TIMEOUT = 3600
DEFAULT_BUILD_TIMEOUT = 1200
DEFAULT_CPUS = 1
DEFAULT_MEMORY_MB = 2048
DEFAULT_STORAGE_MB = 10240


@dataclass(frozen=True)
class PortexTaskRecord:
    task_id: str
    task_prompt: str
    reference_file: str
    criteria: list[dict[str, Any]]
    pass_threshold: float
    tools: list[Any]
    metadata: dict[str, Any]
    environment: dict[str, Any]


def _task_list(tasks_data: Any) -> list[dict[str, Any]]:
    """
    Normalize a Portex tasks structure into a list of task dictionaries.
    
    Accepts either a mapping with a "prompts" key or a list; returns only those items that are dictionaries. If the input is a dict, the function extracts the value of the "prompts" key when it is a list. For any other input shapes or missing/invalid data, an empty list is returned.
    
    Parameters:
        tasks_data (Any): The raw tasks representation — either a dict possibly containing a "prompts" list or a list of prompt entries.
    
    Returns:
        list[dict[str, Any]]: A list containing only the prompt entries that are dictionaries.
    """
    if isinstance(tasks_data, dict):
        prompts = tasks_data.get("prompts")
        if isinstance(prompts, list):
            return [prompt for prompt in prompts if isinstance(prompt, dict)]
        return []
    if isinstance(tasks_data, list):
        return [prompt for prompt in tasks_data if isinstance(prompt, dict)]
    return []


def _load_portex_tasks(bundle_dir: Path) -> list[PortexTaskRecord]:
    """
    Load Portex tasks from a bundle directory and return normalized PortexTaskRecord entries.
    
    Reads tasks.json and answers.json from the provided bundle directory, pairs prompts with their corresponding answers (skipping prompts without a matching answer), and constructs a PortexTaskRecord for each matched task. For each record, the function:
    - extracts the prompt from one of `task_prompt`, `prompt`, or `task`;
    - uses `reference_file` from the prompt if present;
    - filters `criteria` to dict entries;
    - treats `passThreshold` as a float with a default of 100;
    - ensures `tools` is a list (empty if absent);
    - ensures `metadata` and `environment` are dictionaries (empty if absent).
    
    Parameters:
        bundle_dir (Path): Path to the root of a Portex bundle containing `tasks.json` and `answers.json`.
    
    Returns:
        list[PortexTaskRecord]: A list of normalized task records for tasks that have matching answers.
    """
    tasks_data = json.loads((bundle_dir / "tasks.json").read_text(encoding="utf-8"))
    answers_data = json.loads((bundle_dir / "answers.json").read_text(encoding="utf-8"))
    prompts = {item["task_id"]: item for item in _task_list(tasks_data) if "task_id" in item}
    answers = {
        item["task_id"]: item for item in answers_data if isinstance(item, dict) and "task_id" in item
    }

    records: list[PortexTaskRecord] = []
    for task_id, prompt in prompts.items():
        answer = answers.get(task_id)
        if answer is None:
            continue
        records.append(
            PortexTaskRecord(
                task_id=task_id,
                task_prompt=str(prompt.get("task_prompt") or prompt.get("prompt") or prompt.get("task") or ""),
                reference_file=str(prompt.get("reference_file") or ""),
                criteria=[
                    criterion for criterion in answer.get("criteria", []) if isinstance(criterion, dict)
                ],
                pass_threshold=float(answer.get("passThreshold", 100)),
                tools=list(answer.get("tools", [])) if isinstance(answer.get("tools"), list) else [],
                metadata=prompt.get("metadata", {})
                if isinstance(prompt.get("metadata"), dict)
                else {},
                environment=prompt.get("environment", {})
                if isinstance(prompt.get("environment"), dict)
                else {},
            )
        )
    return records


def _write_task_toml(task: PortexTaskRecord, path: Path) -> None:
    """
    Write a Harbor-compatible task.toml for the given PortexTaskRecord to the specified file path.
    
    The generated TOML includes [metadata] (author, email, difficulty, category, tags), [metadata.portex] with task_id, [verifier] and [verifier.env] settings, [agent] timeout, and [environment] settings (base_image, timeouts, CPU/memory/storage and optional GPUs). Missing metadata, environment fields, or timeouts are filled with module defaults; tags always include "portex-eval" and "agent-eval".
    
    Parameters:
        task (PortexTaskRecord): Task record containing metadata, environment, and task_id.
        path (Path): Filesystem path where task.toml will be written (overwrites if exists).
    """
    metadata = task.metadata
    environment = task.environment
    resources = environment.get("resources", {}) if isinstance(environment, dict) else {}
    timeouts = environment.get("timeouts", {}) if isinstance(environment, dict) else {}

    author_name = str(metadata.get("author_name", "Portex Eval"))
    author_email = str(metadata.get("author_email", ""))
    difficulty = str(metadata.get("difficulty", ""))
    category = str(metadata.get("category", ""))
    tags = metadata.get("tags", []) if isinstance(metadata.get("tags"), list) else []
    all_tags = list(dict.fromkeys(["portex-eval", "agent-eval"] + [str(tag) for tag in tags]))

    lines = ['version = "1.0"', "", "[metadata]"]
    lines.append(f'author_name = "{author_name}"')
    if author_email:
        lines.append(f'author_email = "{author_email}"')
    if difficulty:
        lines.append(f'difficulty = "{difficulty}"')
    if category:
        lines.append(f'category = "{category}"')
    tags_str = ", ".join(f'"{tag}"' for tag in all_tags)
    lines.append(f"tags = [{tags_str}]")

    lines.append("")
    lines.append("[metadata.portex]")
    lines.append(f'task_id = "{task.task_id}"')

    lines.append("")
    lines.append("[verifier]")
    lines.append(
        f"timeout_sec = {float(timeouts.get('verifier_sec', DEFAULT_VERIFIER_TIMEOUT))}"
    )
    lines.append("")
    lines.append("[verifier.env]")
    lines.append('OPENROUTER_API_KEY = "${OPENROUTER_API_KEY}"')
    lines.append('PORTEX_JUDGE_MODELS = "${PORTEX_JUDGE_MODELS}"')
    lines.append('PORTEX_JUDGE_CONFIGS = "${PORTEX_JUDGE_CONFIGS}"')

    lines.append("")
    lines.append("[agent]")
    lines.append(f"timeout_sec = {float(timeouts.get('agent_sec', DEFAULT_AGENT_TIMEOUT))}")

    lines.append("")
    lines.append("[environment]")
    lines.append(
        f'base_image = "{environment.get("base_image", DEFAULT_BASE_IMAGE)}"'
    )
    lines.append(f"build_timeout_sec = {float(timeouts.get('build_sec', DEFAULT_BUILD_TIMEOUT))}")
    lines.append(f"cpus = {int(resources.get('cpus', DEFAULT_CPUS))}")
    lines.append(f"memory_mb = {int(resources.get('memory_mb', DEFAULT_MEMORY_MB))}")
    lines.append(f"storage_mb = {int(resources.get('storage_mb', DEFAULT_STORAGE_MB))}")
    gpus = int(resources.get("gpus", 0) or 0)
    if gpus:
        lines.append(f"gpus = {gpus}")

    path.write_text("\n".join(lines), encoding="utf-8")


def _copy_reference_file(bundle_dir: Path, task: PortexTaskRecord, refs_dir: Path) -> None:
    """
    Ensure a task's reference file is placed under the task's refs directory by creating the destination, writing a placeholder, and copying the referenced file from the bundle when present.
    
    Parameters:
        bundle_dir (Path): Root directory of the Portex bundle (expects a "refs" subdirectory).
        task (PortexTaskRecord): Task record whose `reference_file` (and `task_id` for error messages) is used.
        refs_dir (Path): Destination directory where the placeholder and reference file will be placed.
    
    Behavior:
        - Creates `refs_dir` and writes an empty `_placeholder.txt`.
        - If `task.reference_file` is falsy, no further action is taken.
        - If `task.reference_file` is set, copies the file from `bundle_dir/refs/<reference_file>` into `refs_dir`, creating any necessary parent directories.
    
    Raises:
        FileNotFoundError: If `task.reference_file` is specified but the source file does not exist in the bundle's refs directory.
    """
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "_placeholder.txt").write_text("", encoding="utf-8")
    if not task.reference_file:
        return
    src = bundle_dir / "refs" / task.reference_file
    if not src.is_file():
        raise FileNotFoundError(f"Reference file not found for task {task.task_id}: {src}")
    dest = refs_dir / task.reference_file
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _copy_runtime_package(tests_dir: Path) -> None:
    """
    Copy the local runtime package into a task's tests directory under runtime/portex_eval.
    
    Creates tests_dir/runtime if needed and copies the project package (two levels above this file) into tests_dir/runtime/portex_eval, ignoring Python cache directories and .pyc files.
    Parameters:
        tests_dir (Path): Destination tests directory where the runtime package will be placed.
    """
    runtime_dir = tests_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    source_package_dir = Path(__file__).resolve().parents[2]
    shutil.copytree(
        source_package_dir,
        runtime_dir / "portex_eval",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def _prepare_task(bundle_dir: Path, task: PortexTaskRecord, output_dir: Path) -> None:
    """
    Assemble and write all files and directories for a single Portex task into a Harbor-compatible task directory.
    
    Creates output_dir (and subdirectories) and populates it with:
    - task.toml (generated from the task record)
    - instruction.md (template with the task prompt substituted)
    - environment/ (Dockerfile, requirements.txt, and refs copied from the bundle)
    - tests/ (test scripts, grading script, runtime package, task_config.json, criteria.json)
    - solution/ (solve.sh)
    
    Parameters:
        bundle_dir (Path): Path to the source Portex bundle (used to locate reference files).
        task (PortexTaskRecord): Normalized task record containing prompt, reference file name, criteria, thresholds, tools, metadata, and environment.
        output_dir (Path): Target directory to create and populate for this task.
    
    Raises:
        FileNotFoundError: If the task specifies a reference file that does not exist in bundle_dir/refs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_task_toml(task, output_dir / "task.toml")

    instruction = (TEMPLATE_DIR / "instruction.md").read_text(encoding="utf-8")
    (output_dir / "instruction.md").write_text(
        instruction.replace("{task_prompt}", task.task_prompt),
        encoding="utf-8",
    )

    env_dir = output_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    shutil.copy2(TEMPLATE_DIR / "environment" / "Dockerfile", env_dir / "Dockerfile")
    shutil.copy2(
        TEMPLATE_DIR / "environment" / "requirements.txt",
        env_dir / "requirements.txt",
    )
    _copy_reference_file(bundle_dir, task, env_dir / "refs")

    tests_dir = output_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    shutil.copy2(TEMPLATE_DIR / "tests" / "test.sh", tests_dir / "test.sh")
    shutil.copy2(TEMPLATE_DIR / "tests" / "portex_grade.py", tests_dir / "portex_grade.py")
    _copy_runtime_package(tests_dir)

    task_config = {
        "task_id": task.task_id,
        "question": task.task_prompt,
        "reference_file": task.reference_file,
        "submission_path": "/app/answer.txt",
        "pass_threshold": task.pass_threshold,
        "judge_models": list(DEFAULT_AGENT_JUDGE_MODELS),
        "tools": task.tools,
    }
    (tests_dir / "task_config.json").write_text(
        json.dumps(task_config, indent=2),
        encoding="utf-8",
    )
    (tests_dir / "criteria.json").write_text(
        json.dumps(task.criteria, indent=2),
        encoding="utf-8",
    )

    solution_dir = output_dir / "solution"
    solution_dir.mkdir(exist_ok=True)
    shutil.copy2(TEMPLATE_DIR / "solution" / "solve.sh", solution_dir / "solve.sh")


def create_agent_eval_bundle(
    *,
    bundle_dir: str,
    output_dir: str,
    overwrite: bool = False,
) -> AgentEvalBundle:
    """
    Create a Harbor-compatible agent-eval bundle from a Portex bundle.
    
    Creates per-task directories under the output bundle's datasets directory, writes a manifest, and returns an AgentEvalBundle describing the generated bundle.
    
    Parameters:
        bundle_dir (str): Path to the source Portex bundle directory.
        output_dir (str): Path where the agent-eval bundle will be created.
        overwrite (bool): If True, remove any existing output_dir before creating the bundle; if False and output_dir exists and is non-empty, a ValueError is raised.
    
    Returns:
        AgentEvalBundle: Object with attributes `path` (output bundle path), `datasets_dir` (path to the datasets directory), and `task_count` (number of tasks prepared).
    
    Raises:
        FileNotFoundError: If bundle_dir does not exist or is not a directory.
        ValueError: If output_dir exists and is non-empty while overwrite is False.
    """
    source_bundle = Path(bundle_dir).expanduser().resolve()
    if not source_bundle.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {source_bundle}")

    output_root = Path(output_dir).expanduser().resolve()
    datasets_dir = output_root / "datasets"
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise ValueError(
            f"Output directory already exists and is not empty: {output_root}. "
            "Use overwrite=True to allow overwriting."
        )
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    tasks = _load_portex_tasks(source_bundle)
    for task in tasks:
        _prepare_task(source_bundle, task, datasets_dir / f"portex_{task.task_id}")

    manifest = {
        "source_bundle": str(source_bundle),
        "datasets_dir": str(datasets_dir),
        "task_count": len(tasks),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return AgentEvalBundle(
        path=str(output_root),
        datasets_dir=str(datasets_dir),
        task_count=len(tasks),
    )
