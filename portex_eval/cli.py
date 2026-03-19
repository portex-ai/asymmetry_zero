"""CLI entrypoint for portex-eval.

Provides command-line access to the portex_eval API for formatting bundles
and running evaluations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from portex_eval.api import agent_eval as run_agent_eval
from portex_eval.api import create_agent_eval, create_benchmark
from portex_eval.api import eval as run_eval
from portex_eval.errors import PortexEvalError


def _parse_model_config_arg(raw_value: str, flag_name: str) -> dict[str, Any]:
    path = Path(raw_value).expanduser()
    if path.is_file():
        payload_text = path.read_text(encoding="utf-8")
    else:
        payload_text = raw_value

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise PortexEvalError(
            f"{flag_name} must be a JSON object or path to a JSON file: {exc.msg}"
        ) from exc

    if not isinstance(payload, dict):
        raise PortexEvalError(f"{flag_name} must decode to a JSON object.")
    return payload

app = typer.Typer(
    name="portex-eval",
    help="Lightweight evaluation framework for LLM judges and candidates.",
    add_completion=False,
)


def _parse_model_specs(
    models: list[str] | None,
    model_configs: list[str] | None,
    *,
    config_flag_name: str,
) -> list[str | dict[str, Any]]:
    specs: list[str | dict[str, Any]] = list(models or [])
    specs.extend(
        _parse_model_config_arg(raw_value, config_flag_name) for raw_value in (model_configs or [])
    )
    return specs


def _resolve_config_path(base_dir: Path, raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _load_agent_run_config(config_path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PortexEvalError(f"Invalid YAML in --config {config_path}: {exc}") from exc

    if payload is None:
        raise PortexEvalError(f"Config file is empty: {config_path}")
    if not isinstance(payload, dict):
        raise PortexEvalError(f"--config must contain a YAML mapping/object: {config_path}")

    schema_version = payload.get("schema_version", 1)
    if not isinstance(schema_version, int) or schema_version != 1:
        raise PortexEvalError(
            f"Unsupported schema_version={schema_version!r} in {config_path}. "
            "Supported schema_version is 1."
        )

    base_dir = config_path.parent.resolve()
    raw_tasks = payload.get("tasks", payload.get("bundle_path"))
    raw_output = payload.get("output", payload.get("jobs-dir"))
    raw_harbor_args = payload.get("harbor_args", [])
    raw_judges = payload.get("judges")
    raw_judge_configs = payload.get("judge_configs", payload.get("judge-configs"))
    raw_n_concurrent = payload.get("n_concurrent")
    raw_env = payload.get("env")
    raw_overwrite = payload.get("overwrite")

    if raw_harbor_args is None:
        raw_harbor_args = []
    if not isinstance(raw_harbor_args, list) or not all(
        isinstance(item, str) for item in raw_harbor_args
    ):
        raise PortexEvalError("'harbor_args' in --config must be a list of strings.")
    if raw_judges is not None and (
        not isinstance(raw_judges, list) or not all(isinstance(item, str) for item in raw_judges)
    ):
        raise PortexEvalError("'judges' in --config must be a list of strings.")
    if raw_judge_configs is not None and (
        not isinstance(raw_judge_configs, list)
        or not all(isinstance(item, str) for item in raw_judge_configs)
    ):
        raise PortexEvalError("'judge_configs' in --config must be a list of strings.")
    if raw_n_concurrent is not None and not isinstance(raw_n_concurrent, int):
        raise PortexEvalError("'n_concurrent' in --config must be an integer.")
    if raw_env is not None and not isinstance(raw_env, str):
        raise PortexEvalError("'env' in --config must be a string.")
    if raw_overwrite is not None and not isinstance(raw_overwrite, bool):
        raise PortexEvalError("'overwrite' in --config must be a boolean.")

    return {
        "tasks": _resolve_config_path(base_dir, raw_tasks) if isinstance(raw_tasks, str) else None,
        "output": _resolve_config_path(base_dir, raw_output) if isinstance(raw_output, str) else None,
        "judges": list(raw_judges or []),
        "judge_configs": list(raw_judge_configs or []),
        "n_concurrent": raw_n_concurrent,
        "env": raw_env,
        "overwrite": raw_overwrite,
        "harbor_args": list(raw_harbor_args),
    }


@app.command()
def format(
    input_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the benchmark.json input file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
) -> None:
    """Format a benchmark.json file into a Portex bundle.

    Creates a bundle directory with tasks.json, answers.json, and refs/ from
    the input benchmark.json file. The output directory is created next to
    the input file with the same name (minus .json extension).
    """
    try:
        result = create_benchmark(str(input_path))
        typer.echo(f"Created bundle at: {result.path}")
        typer.echo(f"Task count: {result.task_count}")
    except PortexEvalError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
    except Exception as exc:
        typer.secho(f"Unexpected error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None


@app.command()
def run(
    bundle: Annotated[
        Path | None,
        typer.Option(
            "--bundle",
            "-b",
            help="Path to the bundle directory.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ] = None,
    judges: Annotated[
        list[str] | None,
        typer.Option(
            "--judge",
            "-j",
            help="Judge model identifier (e.g., openrouter:openai/gpt-4o). Can be repeated.",
        ),
    ] = None,
    judge_configs: Annotated[
        list[str] | None,
        typer.Option(
            "--judge-config",
            help="Judge model config as JSON or a path to a JSON file. Can be repeated.",
        ),
    ] = None,
    candidates: Annotated[
        list[str] | None,
        typer.Option(
            "--candidate",
            "-c",
            help="Candidate model identifier (e.g., openrouter:openai/gpt-4o). Can be repeated.",
        ),
    ] = None,
    candidate_configs: Annotated[
        list[str] | None,
        typer.Option(
            "--candidate-config",
            help="Candidate model config as JSON or a path to a JSON file. Can be repeated.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for run results.",
        ),
    ] = None,
    task_spec: Annotated[
        str | None,
        typer.Option(
            "--task-spec",
            "-t",
            help="Task specification override.",
        ),
    ] = None,
    max_samples: Annotated[
        int | None,
        typer.Option(
            "--max-samples",
            help="Maximum number of bundle samples to run in parallel.",
            min=1,
        ),
    ] = None,
    logprobs: Annotated[
        bool,
        typer.Option(
            "--logprobs",
            help="Request completion logprobs from the candidate model when supported.",
        ),
    ] = False,
    top_logprobs: Annotated[
        int | None,
        typer.Option(
            "--top-logprobs",
            help="Number of top alternative logprobs to request per completion token.",
            min=1,
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Allow overwriting existing output directories.",
        ),
    ] = False,
) -> None:
    """Run an evaluation benchmark.

    Executes the evaluation with specified judges and candidates, generating
    logs, CSV reports, and reward files.

    Example:
        portex-eval run --bundle ./my-bundle \\
            --judge openrouter/openai/gpt-4o \\
            --candidate openrouter/anthropic/claude-3.5-sonnet
    """
    if bundle is None:
        typer.secho("Error: --bundle is required", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    judge_specs = _parse_model_specs(judges, judge_configs, config_flag_name="--judge-config")
    candidate_specs = _parse_model_specs(
        candidates, candidate_configs, config_flag_name="--candidate-config"
    )

    if not judge_specs:
        typer.secho("Error: At least one --judge is required", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    if not candidate_specs:
        typer.secho("Error: At least one --candidate is required", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    try:
        result = run_eval(
            path=str(bundle),
            judges=judge_specs,
            candidates=candidate_specs,
            output_dir=str(output_dir) if output_dir else None,
            task_spec=task_spec,
            max_samples=max_samples,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            overwrite=overwrite,
        )
        typer.echo(f"Run ID: {result.run_id}")
        typer.echo(f"Output directory: {result.output_dir}")
        typer.echo(f"Logs: {len(result.logs)} file(s)")
        if result.reports:
            typer.echo("Reports generated:")
            typer.echo(f"  - Eval level: {result.reports.eval_level}")
            typer.echo(f"  - Task level: {result.reports.task_level}")
            typer.echo(f"  - Criterion level: {result.reports.criterion_level}")
            typer.echo(f"  - Judgement level: {result.reports.judgement_level}")
        if result.rewards_path:
            typer.echo(f"Rewards JSON: {result.rewards_path}")
            typer.echo(f"Rewards entries: {len(result.rewards.task_ids)}")
        if result.training_data_path:
            typer.echo(f"Training data JSON: {result.training_data_path}")
    except PortexEvalError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
    except Exception as exc:
        typer.secho(f"Unexpected error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None


@app.command("agent-create")
def agent_create(
    bundle: Annotated[
        Path | None,
        typer.Option(
            "--bundle",
            "-b",
            help="Path to the Portex bundle directory.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for generated Harbor tasks.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Allow overwriting an existing Harbor task directory.",
        ),
    ] = False,
) -> None:
    """Convert a Portex bundle into Harbor task directories."""
    if bundle is None:
        typer.secho("Error: --bundle is required", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if output_dir is None:
        typer.secho("Error: --output is required", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    try:
        result = create_agent_eval(
            path=str(bundle),
            output_dir=str(output_dir),
            overwrite=overwrite,
        )
        typer.echo(f"Task root: {result.path}")
        typer.echo(f"Datasets dir: {result.datasets_dir}")
        typer.echo(f"Task count: {result.task_count}")
    except PortexEvalError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
    except Exception as exc:
        typer.secho(f"Unexpected error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None


@app.command(
    "agent-run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def agent_run(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="Path to a YAML config file for agent-run options.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    task_root: Annotated[
        Path | None,
        typer.Option(
            "--tasks",
            "-t",
            help="Path to the generated Harbor task root.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ] = None,
    judges: Annotated[
        list[str] | None,
        typer.Option(
            "--judge",
            "-j",
            help="Judge model identifier (e.g., openrouter:openai/gpt-4o). Can be repeated.",
        ),
    ] = None,
    judge_configs: Annotated[
        list[str] | None,
        typer.Option(
            "--judge-config",
            help="Judge model config as JSON or a path to a JSON file. Can be repeated.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional run output directory. Defaults to the task root.",
        ),
    ] = None,
    n_concurrent: Annotated[
        int | None,
        typer.Option(
            "--n-concurrent",
            help="Maximum number of Harbor tasks to run concurrently.",
            min=1,
        ),
    ] = None,
    env: Annotated[
        str | None,
        typer.Option(
            "--env",
            help="Optional Harbor environment profile name.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Allow overwriting an existing output directory.",
        ),
    ] = False,
) -> None:
    """Run a Harbor-backed agent evaluation."""
    config_payload = _load_agent_run_config(config) if config is not None else {}

    resolved_task_root = task_root or config_payload.get("tasks")
    resolved_output_dir = output_dir or config_payload.get("output")
    resolved_n_concurrent = n_concurrent if n_concurrent is not None else config_payload.get("n_concurrent")
    resolved_env = env or config_payload.get("env")
    resolved_overwrite = overwrite or bool(config_payload.get("overwrite", False))
    resolved_judges = list(config_payload.get("judges", [])) + list(judges or [])
    resolved_judge_configs = list(config_payload.get("judge_configs", [])) + list(judge_configs or [])
    resolved_extra_args = list(config_payload.get("harbor_args", [])) + list(ctx.args)

    if resolved_task_root is None:
        typer.secho("Error: --tasks is required", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    judge_specs = _parse_model_specs(
        resolved_judges,
        resolved_judge_configs,
        config_flag_name="--judge-config",
    )

    try:
        result = run_agent_eval(
            task_root=str(resolved_task_root),
            judges=judge_specs or None,
            output_dir=str(resolved_output_dir) if resolved_output_dir else None,
            n_concurrent=resolved_n_concurrent,
            env=resolved_env,
            extra_args=resolved_extra_args,
            overwrite=resolved_overwrite,
        )
        typer.echo(f"Run ID: {result.run_id}")
        typer.echo(f"Output directory: {result.output_dir}")
        typer.echo(f"Datasets dir: {result.datasets_dir}")
        typer.echo(f"Jobs dir: {result.jobs_dir}")
    except PortexEvalError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
    except Exception as exc:
        typer.secho(f"Unexpected error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None


def main() -> None:
    """Main entry point for the CLI."""
    try:
        app()
    except SystemExit as exc:
        sys.exit(exc.code if exc.code is not None else 0)


if __name__ == "__main__":
    main()
