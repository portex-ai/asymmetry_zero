"""Portex Eval - Lightweight evaluation framework for LLM judges and candidates."""

from __future__ import annotations

from dotenv import load_dotenv

__version__ = "0.1.0"

# Core re-exports - import after __version__ is defined
from portex_eval import reporting as reports
from portex_eval.api import create_benchmark, eval
from portex_eval.config import Config
from portex_eval.errors import PortexEvalError
from portex_eval.run_spec import RunSpec, load_run_spec
from portex_eval.types import Benchmark, EvalResults, ReportPaths

__all__ = [
    "__version__",
    # API
    "create_benchmark",
    "eval",
    # Types
    "Benchmark",
    "EvalResults",
    "ReportPaths",
    "reports",
    "PortexEvalError",
    # Config
    "Config",
    "RunSpec",
    "load_run_spec",
]

load_dotenv()


# Lazy imports for optional subpackages to avoid hard dependencies
def __getattr__(name: str) -> object:
    """Lazy load optional subpackages."""
    if name == "get_provider":
        from portex_eval.providers import get_provider

        return get_provider
    if name == "Provider":
        from portex_eval.providers import Provider

        return Provider
    if name == "Response":
        from portex_eval.providers import Response

        return Response
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
