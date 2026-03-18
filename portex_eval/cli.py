"""CLI entrypoint for portex-eval.

Provides command-line access to the portex_eval API for formatting bundles
and running evaluations.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from portex_eval.api import create_benchmark
from portex_eval.api import eval as run_eval
from portex_eval.errors import PortexEvalError

app = typer.Typer(
    name="portex-eval",
    help="Lightweight evaluation framework for LLM judges and candidates.",
    add_completion=False,
)


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
            help="Judge model identifier (e.g., openrouter/openai/gpt-4o). Can be repeated.",
        ),
    ] = None,
    candidates: Annotated[
        list[str] | None,
        typer.Option(
            "--candidate",
            "-c",
            help="Candidate model identifier (e.g., openrouter/openai/gpt-4o). Can be repeated.",
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

    if not judges:
        typer.secho("Error: At least one --judge is required", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    if not candidates:
        typer.secho("Error: At least one --candidate is required", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    try:
        result = run_eval(
            path=str(bundle),
            judges=list(judges),
            candidates=list(candidates),
            output_dir=str(output_dir) if output_dir else None,
            task_spec=task_spec,
            max_samples=max_samples,
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
