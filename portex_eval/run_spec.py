"""Run specification for portex_eval.

Simplified from portex-bench to minimal fields:
- bundle_path: Path to the evaluation bundle
- judges: List of judge model endpoints
- candidates: List of candidate model endpoints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunSpec:
    """Specification for an evaluation run.

    Attributes:
        bundle_path: Path to the evaluation bundle directory or file.
        judges: List of judge model endpoint identifiers.
        candidates: List of candidate model endpoint identifiers.
        schema_version: Schema version for forward compatibility. Defaults to 1.
    """

    bundle_path: str
    judges: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    schema_version: int = 1

    def __post_init__(self) -> None:
        """Validate the run spec after initialization."""
        if not self.bundle_path:
            raise ValueError("bundle_path is required and cannot be empty")
        if not self.judges:
            raise ValueError("At least one judge model endpoint is required in 'judges'")
        if not self.candidates:
            raise ValueError("At least one candidate model endpoint is required in 'candidates'")

    def validate_bundle_exists(self) -> Path:
        """Validate that the bundle path exists.

        Returns:
            Resolved Path to the bundle.

        Raises:
            FileNotFoundError: If bundle_path does not exist.
        """
        path = Path(self.bundle_path).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Bundle path does not exist: {self.bundle_path!r}. Expected at: {path}"
            )
        return path


def load_run_spec(path: str | Path) -> RunSpec:
    """Load a RunSpec from a YAML file.

    Args:
        path: Path to the YAML run spec file.

    Returns:
        Parsed RunSpec instance.

    Raises:
        FileNotFoundError: If the spec file does not exist.
        ValueError: If the spec file is malformed or missing required fields.
    """
    spec_path = Path(path)
    if not spec_path.is_file():
        raise FileNotFoundError(
            f"Run spec file not found: {path!r}. Expected YAML file at: {spec_path.resolve()}"
        )

    try:
        with spec_path.open("r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in run spec {path!r}: {e}") from e

    if obj is None:
        raise ValueError(f"Run spec file is empty: {path!r}")
    if not isinstance(obj, dict):
        raise ValueError(
            f"Run spec must be a YAML mapping (dict), got {type(obj).__name__}: {path!r}"
        )

    return _parse_run_spec(obj, str(path))


def _parse_run_spec(obj: dict[str, Any], source: str) -> RunSpec:
    """Parse a run spec dictionary into a RunSpec.

    Args:
        obj: Parsed YAML dictionary.
        source: Source path for error messages.

    Returns:
        Parsed RunSpec instance.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    schema_version = obj.get("schema_version", 1)
    if not isinstance(schema_version, int) or schema_version < 1:
        raise ValueError(
            f"schema_version must be a positive integer, got {schema_version!r}: {source}"
        )
    if schema_version != 1:
        raise ValueError(
            f"Unsupported schema_version={schema_version}. "
            f"This version of portex_eval supports schema_version=1: {source}"
        )

    bundle_path = obj.get("bundle_path")
    if not bundle_path:
        raise ValueError(f"Missing required field 'bundle_path' in run spec: {source}")
    if not isinstance(bundle_path, str):
        raise ValueError(
            f"'bundle_path' must be a string, got {type(bundle_path).__name__}: {source}"
        )

    judges = _parse_model_list(obj.get("judges"), "judges", source)
    candidates = _parse_model_list(obj.get("candidates"), "candidates", source)

    return RunSpec(
        bundle_path=bundle_path.strip(),
        judges=judges,
        candidates=candidates,
        schema_version=schema_version,
    )


def _parse_model_list(raw: Any, field_name: str, source: str) -> list[str]:
    """Parse and validate a list of model endpoints.

    Args:
        raw: Raw value from YAML.
        field_name: Field name for error messages.
        source: Source path for error messages.

    Returns:
        List of validated model endpoint strings.

    Raises:
        ValueError: If the field is missing, not a list, or contains invalid entries.
    """
    if raw is None:
        raise ValueError(f"Missing required field '{field_name}' in run spec: {source}")
    if not isinstance(raw, list):
        raise ValueError(f"'{field_name}' must be a list, got {type(raw).__name__}: {source}")
    if not raw:
        raise ValueError(f"'{field_name}' must contain at least one entry: {source}")

    result: list[str] = []
    for i, item in enumerate(raw):
        if not isinstance(item, str):
            raise ValueError(
                f"'{field_name}[{i}]' must be a string, got {type(item).__name__}: {source}"
            )
        stripped = item.strip()
        if not stripped:
            raise ValueError(f"'{field_name}[{i}]' cannot be empty or whitespace-only: {source}")
        result.append(stripped)

    return result


def write_run_spec(path: str | Path, spec: RunSpec) -> None:
    """Write a RunSpec to a YAML file.

    Args:
        path: Destination path for the YAML file.
        spec: RunSpec to serialize.
    """
    spec_path = Path(path)
    spec_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": spec.schema_version,
        "bundle_path": spec.bundle_path,
        "judges": spec.judges,
        "candidates": spec.candidates,
    }

    with spec_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
