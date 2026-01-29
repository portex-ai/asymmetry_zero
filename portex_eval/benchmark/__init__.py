"""Benchmark module for portex_eval.

Provides Inspect AI integration for running LLM evaluations.
"""

from portex_eval.benchmark import inspect
from portex_eval.benchmark.run import (
    BenchmarkMatrixResult,
    BenchmarkResult,
    benchmark_matrix,
    benchmark_one,
)

__all__ = [
    "inspect",
    "benchmark_one",
    "benchmark_matrix",
    "BenchmarkResult",
    "BenchmarkMatrixResult",
]
