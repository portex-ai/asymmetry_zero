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
    """
    Lazily resolve and return selected public attributes from the portex_eval.benchmark package.
    
    This function performs on-demand resolution of exported names so imports occur only when accessed.
    For "inspect" and "harbor" it returns the corresponding submodule; for "benchmark_one", "benchmark_matrix",
    "BenchmarkResult", and "BenchmarkMatrixResult" it returns the matching symbol from portex_eval.benchmark.run.
    
    Parameters:
        name (str): The attribute name being accessed on the package.
    
    Returns:
        object: The resolved module or attribute corresponding to `name`.
    
    Raises:
        AttributeError: If `name` is not one of the supported public attributes.
    """
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
