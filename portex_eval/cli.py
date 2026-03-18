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

from portex_eval.api import agent_eval as run_agent_eval
from portex_eval.api import create_agent_eval, create_benchmark
from portex_eval.api import eval as run_eval
from portex_eval.errors import PortexEvalError


def _parse_model_config_arg(raw_value: str, flag_name: str) -> dict[str, Any]:
    """
    Parse a model configuration from a JSON string or a path to a JSON file.
    
    Parameters:
        raw_value (str): JSON payload text or a filesystem path (string) pointing to a JSON file.
        flag_name (str): CLI flag name used in error messages when parsing fails.
    
    Returns:
        dict[str, Any]: The decoded JSON object representing the model configuration.
    
    Raises:
        PortexEvalError: If the payload is not valid JSON or does not decode to a JSON object.
    """
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
    """
    Combine model identifiers and parsed model configuration entries into a single specifications list.
    
    Parameters:
        models (list[str] | None): Optional list of model identifier strings.
        model_configs (list[str] | None): Optional list of model configuration payloads; each item is either a JSON string or a filesystem path to a JSON file and will be parsed into a dict.
        config_flag_name (str): Flag name used in error messages when parsing model configuration entries.
    
    Returns:
        list[str | dict[str, Any]]: A list where each entry is either a model identifier string or a parsed model configuration dictionary.
    """
    specs: list[str | dict[str, Any]] = list(models or [])
    specs.extend(
        _parse_model_config_arg(raw_value, config_flag_name) for raw_value in (model_configs or [])
    )
    return specs


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
    """
    Convert a Portex bundle into a set of Harbor task directories.
    
    Validates that both bundle and output_dir are provided, then generates Harbor task directories and prints the task root path, datasets directory, and number of tasks created on success.
    
    Parameters:
        bundle (Path | None): Path to the Portex bundle directory (required).
        output_dir (Path | None): Output directory where Harbor tasks will be created (required).
        overwrite (bool): If true, allow overwriting an existing output directory.
    """
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
    """
    Run a Harbor-backed agent evaluation from a Harbor task root.
    
    Parses judge specifications from provided `--judge` and `--judge-config` values, forwards any additional CLI arguments from the Typer context to the evaluation, invokes the agent evaluation, and prints run metadata (run ID, output directory, datasets dir, jobs dir). If present, prints structured report paths and counts for reports, rewards, and training data. On error, prints a descriptive message and exits with code 1.
    
    Parameters:
        ctx: Typer context whose `args` are forwarded to the underlying evaluation as extra arguments.
        task_root: Path to the generated Harbor task root (required).
        judges: One or more judge model identifiers (repeatable).
        judge_configs: One or more judge model configs as JSON strings or paths to JSON files (repeatable); these are merged with `judges` to form judge specifications.
        output_dir: Optional run output directory; defaults to the task root when not provided.
        n_concurrent: Optional maximum number of Harbor tasks to run concurrently (minimum 1).
        env: Optional Harbor environment profile name.
        overwrite: If true, allow overwriting an existing output directory.
    """
    if task_root is None:
        typer.secho("Error: --tasks is required", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    judge_specs = _parse_model_specs(judges, judge_configs, config_flag_name="--judge-config")

    try:
        result = run_agent_eval(
            task_root=str(task_root),
            judges=judge_specs or None,
            output_dir=str(output_dir) if output_dir else None,
            n_concurrent=n_concurrent,
            env=env,
            extra_args=list(ctx.args),
            overwrite=overwrite,
        )
        typer.echo(f"Run ID: {result.run_id}")
        typer.echo(f"Output directory: {result.output_dir}")
        typer.echo(f"Datasets dir: {result.datasets_dir}")
        typer.echo(f"Jobs dir: {result.jobs_dir}")
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


def main() -> None:
    """Main entry point for the CLI."""
    try:
        app()
    except SystemExit as exc:
        sys.exit(exc.code if exc.code is not None else 0)


if __name__ == "__main__":
    main()
