"""Core types for portex_eval programmatic API.

Defines data structures for benchmark configuration, evaluation results,
and output paths used by the API surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Rewards:
    """Reward payload for tasks."""

    task_ids: list[str] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class Benchmark:
    """Minimal benchmark description returned by create_benchmark().

    Attributes:
        path: Absolute path to the generated bundle directory.
        task_count: Number of tasks in the bundle.
    """

    path: str
    task_count: int

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("path is required and cannot be empty")
        if self.task_count < 0:
            raise ValueError("task_count must be non-negative")

    def resolve_path(self) -> Path:
        """Return the absolute Path for the bundle."""
        return Path(self.path).resolve()


@dataclass(frozen=True)
class ReportPaths:
    """Paths to CSV report artifacts from an evaluation run."""

    eval_level: str
    task_level: str
    criterion_level: str
    judgement_level: str


@dataclass(frozen=True)
class EvalResults:
    """Results from a completed evaluation run.

    Attributes:
        logs: Paths to Inspect .log/.eval files.
        reports: Paths to CSV output reports.
        rewards: Reward payload with task_ids and reward list.
        rewards_path: Path to rewards JSON file.
        run_id: Run identifier.
        output_dir: Output directory for this run.
    """

    logs: list[str] = field(default_factory=list)
    reports: ReportPaths | None = None
    rewards: Rewards = field(default_factory=Rewards)
    rewards_path: str = ""
    training_data_path: str = ""
    run_id: str = ""
    output_dir: str = ""

    def with_absolute_paths(self) -> EvalResults:
        """Return a copy with absolute path fields resolved."""
        reports = self.reports
        if reports is not None:
            reports = ReportPaths(
                eval_level=str(Path(reports.eval_level).resolve()),
                task_level=str(Path(reports.task_level).resolve()),
                criterion_level=str(Path(reports.criterion_level).resolve()),
                judgement_level=str(Path(reports.judgement_level).resolve()),
            )
        return EvalResults(
            logs=[str(Path(p).resolve()) for p in self.logs],
            reports=reports,
            rewards=self.rewards,
            rewards_path=str(Path(self.rewards_path).resolve()) if self.rewards_path else "",
            training_data_path=(
                str(Path(self.training_data_path).resolve()) if self.training_data_path else ""
            ),
            run_id=self.run_id,
            output_dir=str(Path(self.output_dir).resolve()) if self.output_dir else "",
        )
