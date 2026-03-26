"""Configuration defaults for portex_eval.

Slim local-first defaults; no lakehouse or remote paths.
Override via environment variables or explicit arguments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Config:
    """Runtime configuration for portex_eval.

    Attributes:
        bundles_dir: Local directory for eval bundles. Defaults to ./bundles.
        runs_dir: Local directory for run outputs. Defaults to ./eval_runs.
        cache_dir: Local cache directory. Defaults to ./.portex_cache.
    """

    bundles_dir: str = field(default_factory=lambda: os.getenv("PORTEX_BUNDLES_DIR", "./bundles"))
    runs_dir: str = field(default_factory=lambda: os.getenv("PORTEX_RUNS_DIR", "./eval_runs"))
    cache_dir: str = field(default_factory=lambda: os.getenv("PORTEX_CACHE_DIR", "./.portex_cache"))

    def resolve_bundle_path(self, bundle_name: str) -> Path:
        """Resolve a bundle name to its absolute path.

        Args:
            bundle_name: Name of the bundle (subdirectory under bundles_dir).

        Returns:
            Absolute Path to the bundle directory.

        Raises:
            FileNotFoundError: If the bundle directory does not exist.
        """
        bundle_path = Path(self.bundles_dir).resolve() / bundle_name
        if not bundle_path.is_dir():
            raise FileNotFoundError(
                f"Bundle not found: {bundle_name!r}. Expected directory at: {bundle_path}"
            )
        return bundle_path

    def ensure_runs_dir(self) -> Path:
        """Ensure runs directory exists and return its path.

        Returns:
            Absolute Path to the runs directory.
        """
        runs_path = Path(self.runs_dir).resolve()
        runs_path.mkdir(parents=True, exist_ok=True)
        return runs_path

    def ensure_cache_dir(self) -> Path:
        """Ensure cache directory exists and return its path.

        Returns:
            Absolute Path to the cache directory.
        """
        cache_path = Path(self.cache_dir).resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    @classmethod
    def from_env(cls) -> Config:
        """Create Config from environment variables.

        Environment variables:
            PORTEX_BUNDLES_DIR: Override bundles_dir
            PORTEX_RUNS_DIR: Override runs_dir
            PORTEX_CACHE_DIR: Override cache_dir

        Returns:
            Config instance with values from environment.
        """
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Create Config from a dictionary.

        Args:
            data: Dictionary with optional keys bundles_dir, runs_dir, cache_dir.

        Returns:
            Config instance.
        """
        return cls(
            bundles_dir=data.get("bundles_dir", os.getenv("PORTEX_BUNDLES_DIR", "./bundles")),
            runs_dir=data.get("runs_dir", os.getenv("PORTEX_RUNS_DIR", "./eval_runs")),
            cache_dir=data.get("cache_dir", os.getenv("PORTEX_CACHE_DIR", "./.portex_cache")),
        )


# Default configuration instance
DEFAULT_CONFIG = Config()
