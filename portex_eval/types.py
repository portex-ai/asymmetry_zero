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
        """
        Resolve the bundle's path to an absolute Path.
        
        Returns:
            Absolute Path pointing to the bundle directory.
        """
        return Path(self.path).resolve()


@dataclass(frozen=True)
class AgentEvalBundle:
    """Generated Harbor task bundle for agentic evals."""

    path: str
    datasets_dir: str
    task_count: int

    def resolve_path(self) -> Path:
        """
        Resolve the absolute Path of the `path` field.
        
        Returns:
            resolved_path (Path): The absolute filesystem Path corresponding to `self.path`.
        """
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
        """
        Produce a copy of this EvalResults with all file- and directory-path fields converted to absolute paths.
        
        Reports (if present) have each subpath resolved to an absolute path; `logs` entries are resolved; `rewards_path`, `training_data_path`, and `output_dir` are resolved when non-empty. `run_id` and `rewards` are preserved unchanged.
        
        Returns:
            EvalResults: A new EvalResults instance with absolute paths for the described fields; fields that were empty strings remain empty strings.
        """
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


@dataclass(frozen=True)
class AgentEvalResults:
    """Results from a completed Harbor-backed agent evaluation run."""

    datasets_dir: str = ""
    jobs_dir: str = ""
    reports: ReportPaths | None = None
    rewards: Rewards = field(default_factory=Rewards)
    rewards_path: str = ""
    training_data_path: str = ""
    run_id: str = ""
    output_dir: str = ""

    def with_absolute_paths(self) -> AgentEvalResults:
        """
        Produce a copy of this AgentEvalResults with all path fields converted to absolute paths.
        
        The returned object preserves non-path fields (e.g., `run_id`, `rewards`) unchanged. For each string path field that is non-empty (`datasets_dir`, `jobs_dir`, `rewards_path`, `training_data_path`, `output_dir`) the value is replaced by its resolved absolute path; empty string fields remain empty. If `reports` is present, each report subpath (`eval_level`, `task_level`, `criterion_level`, `judgement_level`) is resolved to an absolute path in the returned `reports`.
        
        Returns:
            AgentEvalResults: A new AgentEvalResults instance with absolute paths applied to all applicable fields.
        """
        reports = self.reports
        if reports is not None:
            reports = ReportPaths(
                eval_level=str(Path(reports.eval_level).resolve()),
                task_level=str(Path(reports.task_level).resolve()),
                criterion_level=str(Path(reports.criterion_level).resolve()),
                judgement_level=str(Path(reports.judgement_level).resolve()),
            )
        return AgentEvalResults(
            datasets_dir=str(Path(self.datasets_dir).resolve()) if self.datasets_dir else "",
            jobs_dir=str(Path(self.jobs_dir).resolve()) if self.jobs_dir else "",
            reports=reports,
            rewards=self.rewards,
            rewards_path=str(Path(self.rewards_path).resolve()) if self.rewards_path else "",
            training_data_path=(
                str(Path(self.training_data_path).resolve()) if self.training_data_path else ""
            ),
            run_id=self.run_id,
            output_dir=str(Path(self.output_dir).resolve()) if self.output_dir else "",
        )
