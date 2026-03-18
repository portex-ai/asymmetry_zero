"""Benchmark module for portex_eval.

Provides Inspect AI integration for running LLM evaluations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from portex_eval.benchmark import harbor, inspect
    from portex_eval.benchmark.run import (
        BenchmarkMatrixResult,
        BenchmarkResult,
        benchmark_matrix,
        benchmark_one,
    )

__all__ = [
    "inspect",
    "harbor",
    "benchmark_one",
    "benchmark_matrix",
    "BenchmarkResult",
    "BenchmarkMatrixResult",
]


def __getattr__(name: str) -> object:
    if name in {"inspect", "harbor"}:
        import importlib

        return importlib.import_module(f"portex_eval.benchmark.{name}")
    if name in {
        "benchmark_one",
        "benchmark_matrix",
        "BenchmarkResult",
        "BenchmarkMatrixResult",
    }:
        from portex_eval.benchmark import run

        return getattr(run, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
